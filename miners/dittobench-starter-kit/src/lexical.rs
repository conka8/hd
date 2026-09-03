//! Side-car lexical + temporal index over the seeded haystack.
//!
//! The stock kit answers every memory question out of the dense vector
//! pipeline alone (embed -> composite V2 -> cross-encoder rerank). That
//! pipeline is strong on paraphrase and weak on three things DittoBench
//! grades explicitly:
//!
//! 1. **Nonce recall.** A verification code like `VK-8F42` is, to an
//!    embedding model, a bag of meaningless subword pieces. It has no
//!    semantic neighbourhood, so it ranks below topically-similar chatter
//!    and never reaches the model. An exact-match posting list finds it
//!    every time. This is the `canary` category (stock: 0.00) and the
//!    `canary_integrity` composite multiplier.
//! 2. **Rare-term recall.** Proper nouns, product names and coined values
//!    behave the same way, just less extremely. Lexical scoring fused with
//!    the dense ranking recovers them.
//! 3. **Temporal arithmetic.** Order and elapsed time come from the
//!    transcript's own timestamps, not from the model's guess. The index
//!    keeps every pair's timestamp so the injected context can carry dates.
//!
//! It also synthesises a subject index when the validator seeds raw pairs
//! with `subjects: []` (DittoBench "Tier B"), which a prepared-subjects-only
//! harness cannot route.
//!
//! Everything here is per-`user_id`, so multi-graph isolation cases can
//! never read another user's rows out of this index.

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

/// A seeded conversation pair plus the metadata retrieval needs.
#[derive(Clone, Debug)]
pub struct StoredPair {
    pub pair_id: String,
    pub session_id: String,
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    pub prompt: String,
    pub response: String,
    /// Insertion order, used as a stable tiebreak and a recency proxy when
    /// timestamps are absent.
    pub ordinal: usize,
}

impl StoredPair {
    /// Prompt and response as one string, for indexing and extraction.
    pub fn text(&self) -> String {
        format!("{} {}", self.prompt, self.response)
    }
}

/// A scored lexical hit.
#[derive(Clone, Debug)]
pub struct Hit {
    pub pair: StoredPair,
    pub score: f32,
    /// True when the hit was found by exact match on a nonce-shaped token
    /// (a verification code, a coined identifier). These are promoted ahead
    /// of ordinary BM25 hits because embeddings systematically miss them.
    pub exact_nonce: bool,
}

// ---------------------------------------------------------------------------
// tokenisation
// ---------------------------------------------------------------------------

/// English function words. Dropped from postings so the index stays small and
/// BM25 is not dominated by glue.
const STOP: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has",
    "had", "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her",
    "it", "its", "we", "us", "our", "they", "them", "their", "to", "of", "in", "on", "at", "for",
    "with", "about", "as", "by", "from", "into", "over", "so", "just", "very", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must", "not", "no", "yes", "what",
    "when", "where", "who", "whom", "which", "how", "why", "there", "here", "some", "any", "all",
];

/// Shortest token allowed to link two rows.
///
/// Two characters, because names are the strongest links this corpus has and
/// several of them are two characters long. Rarity does the filtering: a term
/// is only followed when it appears in few enough rows to identify something.
const MIN_LINK_TERM_CHARS: usize = 2;

fn is_stop(tok: &str) -> bool {
    STOP.binary_search(&tok).is_ok() || STOP.contains(&tok)
}

/// Splits text into lowercase tokens. Runs of alphanumerics are kept whole,
/// and an internal hyphen or underscore between alphanumerics is preserved so
/// codes like `VK-8F42` survive as one token instead of shattering into
/// `vk` + `8` + `f42`.
pub fn tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c.is_alphanumeric() {
            cur.push(c.to_ascii_lowercase());
        } else if (c == '-' || c == '_')
            && !cur.is_empty()
            && i + 1 < chars.len()
            && chars[i + 1].is_alphanumeric()
        {
            // Internal separator inside a code: keep it.
            cur.push('-');
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        i += 1;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// True for tokens embeddings represent badly: mixed letters+digits, or a
/// hyphenated code, or a long all-digit run. These are the ones an exact
/// index has to catch because the dense pipeline will not.
pub fn nonce_like(tok: &str) -> bool {
    if tok.len() < 3 {
        return false;
    }
    let has_alpha = tok.chars().any(|c| c.is_ascii_alphabetic());
    let has_digit = tok.chars().any(|c| c.is_ascii_digit());
    let has_sep = tok.contains('-');
    if has_alpha && has_digit {
        return true;
    }
    if has_sep && tok.len() >= 5 {
        return true;
    }
    if !has_alpha && has_digit && tok.len() >= 4 {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// per-user index
// ---------------------------------------------------------------------------

#[derive(Default)]
struct UserIndex {
    pairs: Vec<StoredPair>,
    by_id: HashMap<String, usize>,
    /// token -> pair indices (deduplicated, ascending)
    postings: HashMap<String, Vec<u32>>,
    /// token -> total occurrences across the corpus
    term_freq: HashMap<String, HashMap<u32, u32>>,
    doc_len: Vec<u32>,
    total_len: u64,
}

impl UserIndex {
    fn upsert(&mut self, pair: StoredPair) {
        // Idempotent: staged waves re-send pairs, and the wire contract says
        // seeding is an ordered upsert. Replacing in place keeps the postings
        // consistent without a full rebuild for the common no-op case.
        if let Some(&idx) = self.by_id.get(&pair.pair_id) {
            if self.pairs[idx].prompt == pair.prompt && self.pairs[idx].response == pair.response {
                self.pairs[idx].timestamp = pair.timestamp.or(self.pairs[idx].timestamp);
                return;
            }
            self.remove_postings(idx as u32);
            self.pairs[idx] = pair;
            self.add_postings(idx as u32);
            return;
        }
        let idx = self.pairs.len() as u32;
        let mut pair = pair;
        pair.ordinal = idx as usize;
        self.by_id.insert(pair.pair_id.clone(), idx as usize);
        self.pairs.push(pair);
        self.doc_len.push(0);
        self.add_postings(idx);
    }

    fn add_postings(&mut self, idx: u32) {
        let toks = tokenize(&self.pairs[idx as usize].text());
        let mut counts: HashMap<String, u32> = HashMap::new();
        for t in &toks {
            if is_stop(t) {
                continue;
            }
            *counts.entry(t.clone()).or_insert(0) += 1;
        }
        let len = toks.len() as u32;
        self.doc_len[idx as usize] = len;
        self.total_len += len as u64;
        for (tok, n) in counts {
            let posting = self.postings.entry(tok.clone()).or_default();
            if let Err(pos) = posting.binary_search(&idx) {
                posting.insert(pos, idx);
            }
            self.term_freq.entry(tok).or_default().insert(idx, n);
        }
    }

    fn remove_postings(&mut self, idx: u32) {
        self.total_len = self
            .total_len
            .saturating_sub(self.doc_len[idx as usize] as u64);
        self.doc_len[idx as usize] = 0;
        for (_, posting) in self.postings.iter_mut() {
            if let Ok(pos) = posting.binary_search(&idx) {
                posting.remove(pos);
            }
        }
        for (_, tf) in self.term_freq.iter_mut() {
            tf.remove(&idx);
        }
    }

    fn avg_len(&self) -> f32 {
        if self.pairs.is_empty() {
            return 1.0;
        }
        (self.total_len as f32 / self.pairs.len() as f32).max(1.0)
    }

    /// Okapi BM25 with the usual k1/b, plus a multiplier on nonce-shaped
    /// query terms so an exact code match outranks any amount of topical
    /// overlap.
    fn bm25(&self, query: &str, k: usize) -> Vec<Hit> {
        const K1: f32 = 1.4;
        const B: f32 = 0.72;
        /// A nonce match is categorically more informative than a word match:
        /// nothing else in the corpus looks like it.
        const NONCE_BOOST: f32 = 6.0;

        let n = self.pairs.len() as f32;
        if n == 0.0 {
            return Vec::new();
        }
        let avg = self.avg_len();
        let mut seen: HashSet<String> = HashSet::new();
        let mut scores: HashMap<u32, f32> = HashMap::new();
        let mut nonce_hit: HashSet<u32> = HashSet::new();

        for tok in tokenize(query) {
            if is_stop(&tok) || !seen.insert(tok.clone()) {
                continue;
            }
            let Some(posting) = self.postings.get(&tok) else {
                continue;
            };
            let df = posting.len() as f32;
            if df == 0.0 {
                continue;
            }
            let idf = (((n - df + 0.5) / (df + 0.5)) + 1.0).ln();
            let boost = if nonce_like(&tok) { NONCE_BOOST } else { 1.0 };
            for &idx in posting {
                let tf = self
                    .term_freq
                    .get(&tok)
                    .and_then(|m| m.get(&idx))
                    .copied()
                    .unwrap_or(0) as f32;
                let dl = self.doc_len[idx as usize] as f32;
                let denom = tf + K1 * (1.0 - B + B * dl / avg);
                if denom <= 0.0 {
                    continue;
                }
                *scores.entry(idx).or_insert(0.0) += boost * idf * (tf * (K1 + 1.0)) / denom;
                if boost > 1.0 {
                    nonce_hit.insert(idx);
                }
            }
        }

        let mut hits: Vec<Hit> = scores
            .into_iter()
            .map(|(idx, score)| Hit {
                pair: self.pairs[idx as usize].clone(),
                score,
                exact_nonce: nonce_hit.contains(&idx),
            })
            .collect();
        // Exact-nonce hits first, then score, then most recent. The final
        // tiebreak matters for knowledge-update cases, where the latest
        // statement of a value is the correct one.
        hits.sort_by(|a, b| {
            b.exact_nonce
                .cmp(&a.exact_nonce)
                .then(
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(b.pair.ordinal.cmp(&a.pair.ordinal))
        });
        hits.truncate(k);
        hits
    }
}

// ---------------------------------------------------------------------------
// public index
// ---------------------------------------------------------------------------

/// Thread-safe lexical index keyed by `user_id`.
#[derive(Default)]
pub struct LexicalIndex {
    users: RwLock<HashMap<String, UserIndex>>,
}

impl LexicalIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or updates one pair for a user. Called from `/seed`.
    pub fn upsert(&self, user_id: &str, pair: StoredPair) {
        let mut guard = self.users.write().expect("lexical index poisoned");
        guard.entry(user_id.to_string()).or_default().upsert(pair);
    }

    /// Number of pairs indexed for a user.
    pub fn len(&self, user_id: &str) -> usize {
        self.users
            .read()
            .expect("lexical index poisoned")
            .get(user_id)
            .map(|u| u.pairs.len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self, user_id: &str) -> bool {
        self.len(user_id) == 0
    }

    /// Best lexical matches for `query` within one user's graph.
    pub fn search(&self, user_id: &str, query: &str, k: usize) -> Vec<Hit> {
        self.users
            .read()
            .expect("lexical index poisoned")
            .get(user_id)
            .map(|u| u.bm25(query, k))
            .unwrap_or_default()
    }

    /// Every pair whose text contains one of the nonce-shaped tokens in
    /// `query`, regardless of BM25 rank. Used for the canary category, where
    /// the question names the code's *kind* ("verification code") rather than
    /// the code itself, so the reverse also matters: see [`Self::nonce_rows`].
    pub fn exact_nonce_matches(&self, user_id: &str, query: &str) -> Vec<Hit> {
        let guard = self.users.read().expect("lexical index poisoned");
        let Some(u) = guard.get(user_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for tok in tokenize(query).into_iter().filter(|t| nonce_like(t)) {
            let Some(posting) = u.postings.get(&tok) else {
                continue;
            };
            for &idx in posting {
                if seen.insert(idx) {
                    out.push(Hit {
                        pair: u.pairs[idx as usize].clone(),
                        score: f32::MAX,
                        exact_nonce: true,
                    });
                }
            }
        }
        out
    }

    /// Every pair that *contains* a nonce-shaped token, most recent first.
    ///
    /// A canary question ("what is my verification code for this session?")
    /// carries no nonce itself, so exact lookup cannot start from the query.
    /// The set of pairs holding any code is small, and handing all of them to
    /// the model lets it pick the one attributed to this user and reject the
    /// decoy attributed to somebody else. Capped so it can never become a
    /// whole-store dump.
    pub fn nonce_rows(&self, user_id: &str, cap: usize) -> Vec<Hit> {
        let guard = self.users.read().expect("lexical index poisoned");
        let Some(u) = guard.get(user_id) else {
            return Vec::new();
        };
        let mut out: Vec<Hit> = u
            .pairs
            .iter()
            .filter(|p| tokenize(&p.text()).iter().any(|t| nonce_like(t)))
            .map(|p| Hit {
                pair: p.clone(),
                score: 0.0,
                exact_nonce: true,
            })
            .collect();
        out.sort_by(|a, b| b.pair.ordinal.cmp(&a.pair.ordinal));
        out.truncate(cap);
        out
    }

    /// Pairs matched by the *rare* terms in a query, most informative first.
    ///
    /// The generator coins per-run vocabulary ("tavielle", "orinora") and
    /// defines it inside the seeded history, then asks questions that use
    /// those words. Such a term is not nonce-shaped, so
    /// [`Self::exact_nonce_matches`] never sees it, and BM25 dilutes it among
    /// ordinary words. But it is exactly the term whose *definition* the
    /// answer depends on.
    ///
    /// A term is rare when it occurs in at most `max_df` pairs. Those pairs
    /// are returned whole, so the row defining the word travels with the rows
    /// using it. This is the first hop of a two-hop lookup: resolve the
    /// vocabulary, then read the values.
    pub fn rare_term_matches(&self, user_id: &str, query: &str, max_df: usize, cap: usize) -> Vec<Hit> {
        let guard = self.users.read().expect("lexical index poisoned");
        let Some(u) = guard.get(user_id) else {
            return Vec::new();
        };
        let mut terms: Vec<(&String, &Vec<u32>)> = tokenize(query)
            .into_iter()
            // Rarity, not length, decides whether a term identifies anything.
            // A four-character floor was standing in for rarity and throwing
            // away the most identifying tokens in the corpus: people are
            // referred to by nicknames like "Red", "Em", "Mo", "Ace" and
            // "Pip", and the row that binds a nickname to a person was
            // reaching the model in almost no case as a result. The document
            // frequency filter below is the real noise control, and a common
            // short word cannot survive it.
            .filter(|t| t.len() >= MIN_LINK_TERM_CHARS && !is_stop(t))
            .filter_map(|t| u.postings.get_key_value(&t))
            .filter(|(_, posting)| !posting.is_empty() && posting.len() <= max_df)
            .collect();
        // Rarest first: the fewer pairs a term touches, the more it pins down.
        terms.sort_by_key(|(_, posting)| posting.len());

        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for (_, posting) in terms {
            for &idx in posting {
                if out.len() >= cap {
                    return out;
                }
                if seen.insert(idx) {
                    out.push(Hit {
                        pair: u.pairs[idx as usize].clone(),
                        score: f32::MAX,
                        exact_nonce: false,
                    });
                }
            }
        }
        out
    }

    /// Follows the chain: rare terms found *inside already-retrieved rows*
    /// that the question never mentioned, used to retrieve one hop further.
    ///
    /// Many questions are entity-resolution chains. "Resolve the internal
    /// owner for the retail launch, then use their current email" names a
    /// project, and the answer needs the owner's name, their employer change
    /// and their new address. Those rows share no vocabulary with the
    /// question, so no single search can reach them however deep it goes.
    /// The only route is to read what came back, notice the names it
    /// introduced, and search again for those.
    ///
    /// `known` holds terms already used (from the question and earlier hops)
    /// so each round follows genuinely new leads instead of re-treading.
    /// Terms are ranked rarest-first, which favours specific identities over
    /// incidental vocabulary.
    pub fn expand_from(
        &self,
        user_id: &str,
        seeds: &[Hit],
        known: &mut HashSet<String>,
        max_df: usize,
        cap: usize,
    ) -> Vec<Hit> {
        let guard = self.users.read().expect("lexical index poisoned");
        let Some(u) = guard.get(user_id) else {
            return Vec::new();
        };
        let mut leads: Vec<(&String, &Vec<u32>)> = Vec::new();
        let mut considered: HashSet<String> = HashSet::new();
        for hit in seeds {
            for tok in tokenize(&hit.pair.text()) {
                if tok.len() < MIN_LINK_TERM_CHARS || is_stop(&tok) || known.contains(&tok) {
                    continue;
                }
                if !considered.insert(tok.clone()) {
                    continue;
                }
                if let Some((k, posting)) = u.postings.get_key_value(&tok) {
                    if !posting.is_empty() && posting.len() <= max_df {
                        leads.push((k, posting));
                    }
                }
            }
        }
        leads.sort_by(|a, b| a.1.len().cmp(&b.1.len()).then(a.0.cmp(b.0)));

        let already: HashSet<&str> = seeds.iter().map(|h| h.pair.pair_id.as_str()).collect();
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for (term, posting) in leads {
            if out.len() >= cap {
                break;
            }
            known.insert(term.clone());
            for &idx in posting {
                if out.len() >= cap {
                    break;
                }
                let p = &u.pairs[idx as usize];
                if already.contains(p.pair_id.as_str()) || !seen.insert(idx) {
                    continue;
                }
                out.push(Hit { pair: p.clone(), score: 0.0, exact_nonce: false });
            }
        }
        out
    }

    /// Synthesises subject labels from the corpus when the validator seeded
    /// raw pairs with no subject graph (DittoBench "Tier B").
    ///
    /// Deliberately cheap: the most frequent non-stop content terms that
    /// appear in more than one session, which is a good proxy for "a topic
    /// this user returns to". Returns `(label, pair_ids)`.
    pub fn synthesize_subjects(&self, user_id: &str, max_subjects: usize) -> Vec<(String, Vec<String>)> {
        let guard = self.users.read().expect("lexical index poisoned");
        let Some(u) = guard.get(user_id) else {
            return Vec::new();
        };
        let mut ranked: Vec<(&String, &Vec<u32>)> = u
            .postings
            .iter()
            .filter(|(tok, posting)| {
                // A topic recurs across pairs but is not ubiquitous, and is a
                // real word rather than a code or a number.
                posting.len() >= 2
                    && posting.len() <= (u.pairs.len() / 2).max(3)
                    && tok.len() >= 4
                    && !nonce_like(tok)
                    && tok.chars().all(|c| c.is_ascii_alphabetic())
            })
            .collect();
        ranked.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));
        ranked.truncate(max_subjects);
        ranked
            .into_iter()
            .map(|(tok, posting)| {
                (
                    tok.clone(),
                    posting
                        .iter()
                        .map(|&i| u.pairs[i as usize].pair_id.clone())
                        .collect(),
                )
            })
            .collect()
    }

    /// The user's pairs in chronological order, oldest first, capped.
    /// Timestamp-bearing pairs sort by timestamp; the rest fall back to seed
    /// order, which the validator sends chronologically.
    pub fn chronology(&self, user_id: &str, cap: usize) -> Vec<StoredPair> {
        let guard = self.users.read().expect("lexical index poisoned");
        let Some(u) = guard.get(user_id) else {
            return Vec::new();
        };
        let mut pairs = u.pairs.clone();
        pairs.sort_by(|a, b| match (a.timestamp, b.timestamp) {
            (Some(x), Some(y)) => x.cmp(&y),
            _ => a.ordinal.cmp(&b.ordinal),
        });
        if pairs.len() > cap {
            // Keep the newest `cap`: temporal questions overwhelmingly ask
            // about recent state ("most recent", "before the last change").
            pairs = pairs.split_off(pairs.len() - cap);
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(id: &str, prompt: &str, response: &str) -> StoredPair {
        StoredPair {
            pair_id: id.to_string(),
            session_id: "s".to_string(),
            timestamp: None,
            prompt: prompt.to_string(),
            response: response.to_string(),
            ordinal: 0,
        }
    }

    #[test]
    fn keeps_hyphenated_codes_whole() {
        assert_eq!(tokenize("code VK-8F42 ok"), vec!["code", "vk-8f42", "ok"]);
    }

    #[test]
    fn recognises_nonce_shapes() {
        assert!(nonce_like("vk-8f42"));
        assert!(nonce_like("a1b2c3"));
        assert!(nonce_like("2026"));
        assert!(!nonce_like("kayak"));
        assert!(!nonce_like("ab"));
    }

    #[test]
    fn exact_code_outranks_topical_chatter() {
        let ix = LexicalIndex::new();
        for i in 0..40 {
            ix.upsert(
                "u",
                pair(
                    &format!("noise-{i}"),
                    "we talked about the verification process at length",
                    "yes the verification process is important",
                ),
            );
        }
        ix.upsert(
            "u",
            pair("needle", "my session code is VK-8F42", "noted, VK-8F42"),
        );
        let hits = ix.search("u", "VK-8F42", 5);
        assert_eq!(hits[0].pair.pair_id, "needle");
        assert!(hits[0].exact_nonce);
    }

    #[test]
    fn isolation_never_crosses_users() {
        let ix = LexicalIndex::new();
        ix.upsert("alice", pair("a1", "my code is VK-1111", "ok"));
        ix.upsert("bob", pair("b1", "my code is VK-2222", "ok"));
        let hits = ix.search("bob", "VK-1111", 5);
        assert!(hits.is_empty(), "bob must not see alice's rows");
        assert_eq!(ix.len("alice"), 1);
    }

    #[test]
    fn upsert_is_idempotent_across_waves() {
        let ix = LexicalIndex::new();
        ix.upsert("u", pair("p1", "hello", "world"));
        ix.upsert("u", pair("p1", "hello", "world"));
        assert_eq!(ix.len("u"), 1);
    }

    #[test]
    fn nonce_rows_collects_code_bearing_pairs_only() {
        let ix = LexicalIndex::new();
        ix.upsert("u", pair("p1", "I like kayaking", "nice"));
        ix.upsert("u", pair("p2", "my code is VK-8F42", "ok"));
        let rows = ix.nonce_rows("u", 10);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].pair.pair_id, "p2");
    }

    #[test]
    fn rare_terms_pull_in_their_defining_row() {
        let ix = LexicalIndex::new();
        for i in 0..60 {
            ix.upsert("u", pair(&format!("f{i}"), "the quarterly balance was reviewed", "noted"));
        }
        ix.upsert("u", pair("def", "field conventions: tavielle is the settled amount", "on record"));
        ix.upsert("u", pair("val", "the tavielle came to 1622063 this cycle", "logged"));
        let hits = ix.rare_term_matches("u", "which balance remains under tavielle?", 20, 10);
        let ids: Vec<&str> = hits.iter().map(|h| h.pair.pair_id.as_str()).collect();
        assert!(ids.contains(&"def"), "the DEFINITION row must come back: {ids:?}");
        assert!(ids.contains(&"val"), "the value row must come back: {ids:?}");
    }

    #[test]
    fn rare_terms_ignore_ubiquitous_words() {
        let ix = LexicalIndex::new();
        for i in 0..40 {
            ix.upsert("u", pair(&format!("f{i}"), "balance review meeting", "noted"));
        }
        // "balance" is in every pair, so it pins nothing down and must not
        // drag the whole corpus into the prompt.
        let hits = ix.rare_term_matches("u", "what was the balance?", 5, 50);
        assert!(hits.is_empty(), "a ubiquitous term must not match, got {}", hits.len());
    }

    #[test]
    fn expansion_follows_an_entity_chain_the_question_never_names() {
        let ix = LexicalIndex::new();
        for i in 0..50 {
            ix.upsert("u", pair(&format!("f{i}"), "routine status update on the account", "noted"));
        }
        // The question can only reach ev0. Everything after it is reachable
        // only by reading ev0 and following the name it introduces.
        ix.upsert("u", pair("ev0", "the kestrel rollout means the Faircroft account, owner Lackey", "understood"));
        ix.upsert("u", pair("ev1", "Lackey moved to Ravenwyn Company last spring", "noted"));
        // Deliberately shares no vocabulary with the question: that is the
        // whole point, and an earlier version of this fixture leaked the word
        // "address" into both, letting one pass find it and quietly proving
        // nothing.
        ix.upsert("u", pair("ev2", "since the Ravenwyn move the email is m.lackey@ravenwyn.com", "saved"));

        let q = "for the kestrel rollout, who should I contact?";
        let first = ix.search("u", q, 6);
        assert!(first.iter().any(|h| h.pair.pair_id == "ev0"));
        assert!(!first.iter().any(|h| h.pair.pair_id == "ev2"),
            "one pass cannot reach the answer row");

        let mut known: HashSet<String> = tokenize(q).into_iter().collect();
        let hop1 = ix.expand_from("u", &first, &mut known, 12, 20);
        let mut all = first.clone();
        all.extend(hop1.clone());
        let hop2 = ix.expand_from("u", &hop1, &mut known, 12, 20);
        all.extend(hop2);

        let ids: Vec<&str> = all.iter().map(|h| h.pair.pair_id.as_str()).collect();
        assert!(ids.contains(&"ev2"), "expansion must reach the answer row: {ids:?}");
    }
}
