//! Entity-graph retrieval by Personalized PageRank.
//!
//! Follows HippoRAG (arXiv:2405.14831), which reports single-step retrieval
//! matching or beating iterative methods like IRCoT on multi-hop QA while
//! being 10-20x cheaper and 6-13x faster. That cost ratio is the whole reason
//! this module exists: a hand-rolled iterative expander was measured here to
//! raise evidence recall from 8% to 46% and simultaneously drop the composite
//! from 0.304 to 0.152, because the extra passes pushed the hardest cases
//! past their deadline and they returned nothing.
//!
//! The problem it solves is structural, not a tuning failure. Consider a real
//! generated question:
//!
//! > "I am sending the 'kestrel flight rollout' update. Resolve the internal
//! > owner from the retail launch work for Faircroft Collective, then use
//! > their current rather than original email."
//!
//! The answer lives in a memory that says none of those words. It is reached
//! only through: project alias -> the row naming the owner -> that person's
//! employer change -> the address that followed the move. Scoring passages
//! against the question cannot get there at any depth, because relevance is
//! measured against the wrong thing.
//!
//! A graph gets there in one pass. Passages and the entities they mention
//! form a bipartite graph. Mass starts on the entities the question actually
//! names, then flows entity -> passage -> entity repeatedly. Each sweep is
//! one hop, so the owner's email accumulates mass through the chain without
//! anyone issuing a second query. Cost is a few sparse matrix sweeps: sub-
//! millisecond, and flat in the number of hops.
//!
//! Entity extraction here is deliberately LLM-free. HippoRAG uses an LLM for
//! open information extraction at index time; this harness cannot, because
//! `/seed` is bounded at five minutes per wave and no model is guaranteed to
//! be reachable. Proper-noun runs, quoted spans, addresses and coined codes
//! recover most of what matters in this corpus, and they cost nothing.

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use crate::lexical::{nonce_like, StoredPair};

/// Damping: the share of mass that keeps walking rather than restarting at
/// the question's own entities. Higher reaches further along a chain but
/// blurs toward corpus-wide popularity.
const DAMPING: f32 = 0.55;

/// Sweeps of entity -> passage -> entity. Each is one hop, and the observed
/// chains run to four links.
const SWEEPS: usize = 5;

/// An entity mentioned in more than this share of passages carries no
/// information and only spreads mass indiscriminately.
const UBIQUITY_RATIO: f32 = 0.18;

/// A scored passage.
#[derive(Clone, Debug)]
pub struct Ranked {
    pub pair: StoredPair,
    pub score: f32,
}

#[derive(Default)]
struct UserGraph {
    pairs: Vec<StoredPair>,
    by_id: HashMap<String, usize>,
    entity_id: HashMap<String, u32>,
    /// entity -> passages mentioning it
    ent_pairs: Vec<Vec<u32>>,
    /// passage -> entities it mentions
    pair_ents: Vec<Vec<u32>>,
}

impl UserGraph {
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.entity_id.get(name) {
            return id;
        }
        let id = self.ent_pairs.len() as u32;
        self.entity_id.insert(name.to_string(), id);
        self.ent_pairs.push(Vec::new());
        id
    }

    fn upsert(&mut self, pair: StoredPair) {
        if let Some(&idx) = self.by_id.get(&pair.pair_id) {
            // Staged waves re-send pairs unchanged; rebuilding the edges for
            // an identical row would only churn.
            if self.pairs[idx].prompt == pair.prompt && self.pairs[idx].response == pair.response {
                return;
            }
        }
        let text = format!("{} {}", pair.prompt, pair.response);
        let idx = match self.by_id.get(&pair.pair_id) {
            Some(&i) => {
                self.pairs[i] = pair;
                i
            }
            None => {
                self.by_id.insert(pair.pair_id.clone(), self.pairs.len());
                self.pairs.push(pair);
                self.pair_ents.push(Vec::new());
                self.pairs.len() - 1
            }
        };
        let ents: Vec<String> = extract_entities(&text);
        let mut ids = Vec::with_capacity(ents.len());
        for name in ents {
            let eid = self.intern(&name);
            if !self.ent_pairs[eid as usize].contains(&(idx as u32)) {
                self.ent_pairs[eid as usize].push(idx as u32);
            }
            if !ids.contains(&eid) {
                ids.push(eid);
            }
        }
        self.pair_ents[idx] = ids;
    }

    /// Personalized PageRank seeded on the question's entities, scoring
    /// passages by the mass that reaches them.
    fn rank(&self, query: &str, k: usize) -> Vec<Ranked> {
        if self.pairs.is_empty() {
            return Vec::new();
        }
        let ubiquity = ((self.pairs.len() as f32) * UBIQUITY_RATIO).max(2.0) as usize;

        // Seed on entities the question actually names.
        let mut seed = vec![0.0f32; self.ent_pairs.len()];
        let mut seeded = 0usize;
        for name in extract_entities(query) {
            if let Some(&eid) = self.entity_id.get(&name) {
                if self.ent_pairs[eid as usize].len() <= ubiquity {
                    seed[eid as usize] += 1.0;
                    seeded += 1;
                }
            }
        }
        if seeded == 0 {
            return Vec::new();
        }
        let norm: f32 = seed.iter().sum();
        for s in seed.iter_mut() {
            *s /= norm;
        }

        let mut ent_mass = seed.clone();
        let mut pair_mass = vec![0.0f32; self.pairs.len()];
        let mut acc = vec![0.0f32; self.pairs.len()];

        for _ in 0..SWEEPS {
            // entity -> passage
            for m in pair_mass.iter_mut() {
                *m = 0.0;
            }
            for (eid, mass) in ent_mass.iter().enumerate() {
                if *mass <= f32::EPSILON {
                    continue;
                }
                let targets = &self.ent_pairs[eid];
                if targets.is_empty() || targets.len() > ubiquity {
                    continue;
                }
                let share = mass / targets.len() as f32;
                for &p in targets {
                    pair_mass[p as usize] += share;
                }
            }
            for (i, m) in pair_mass.iter().enumerate() {
                acc[i] += *m;
            }
            // passage -> entity, then restart a share of the mass at the seed
            let mut next = vec![0.0f32; self.ent_pairs.len()];
            for (pid, mass) in pair_mass.iter().enumerate() {
                if *mass <= f32::EPSILON {
                    continue;
                }
                let ents = &self.pair_ents[pid];
                if ents.is_empty() {
                    continue;
                }
                let share = mass / ents.len() as f32;
                for &e in ents {
                    next[e as usize] += share;
                }
            }
            for (e, m) in ent_mass.iter_mut().enumerate() {
                *m = DAMPING * next[e] + (1.0 - DAMPING) * seed[e];
            }
        }

        let mut out: Vec<Ranked> = acc
            .iter()
            .enumerate()
            .filter(|(_, s)| **s > 0.0)
            .map(|(i, s)| Ranked { pair: self.pairs[i].clone(), score: *s })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.pair.ordinal.cmp(&a.pair.ordinal))
        });
        out.truncate(k);
        out
    }
}

/// Names worth linking passages by: proper-noun runs, quoted spans, email
/// addresses and coined codes. Lowercased so "Micky" and "micky" are one node.
///
/// Sentence-initial words are the known false positive of capitalisation
/// heuristics, so a single capitalised token that opens a sentence is only
/// kept when it is not ordinary vocabulary.
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut push = |s: String, out: &mut Vec<String>, seen: &mut HashSet<String>| {
        let s = s.trim().to_lowercase();
        if s.len() >= 3 && seen.insert(s.clone()) {
            out.push(s);
        }
    };
    // A multi-word name is emitted whole AND in parts. Two passages naming
    // the same person do not agree on the span: one writes "owner Michael
    // Lackey" mid-sentence, the next opens with "Michael Lackey used to...".
    // Whole-span-only extraction makes those different nodes and the chain
    // silently breaks, which is exactly how this failed the first time.
    let mut push_run = |words: &[String], out: &mut Vec<String>, seen: &mut HashSet<String>| {
        if words.is_empty() {
            return;
        }
        if words.len() > 1 {
            push(words.join(" "), out, seen);
        }
        for w in words {
            push(w.clone(), out, seen);
        }
    };

    // Quoted spans: the generator marks coined project names this way.
    let bytes: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if c == '"' || c == '\u{201c}' || c == '\u{2018}' || c == '\'' {
            let close = match c {
                '\u{201c}' => '\u{201d}',
                '\u{2018}' => '\u{2019}',
                other => other,
            };
            if let Some(end) = (i + 1..bytes.len().min(i + 90)).find(|&j| bytes[j] == close) {
                let span: String = bytes[i + 1..end].iter().collect();
                if span.split_whitespace().count() <= 6 {
                    push(span, &mut out, &mut seen);
                }
                i = end + 1;
                continue;
            }
        }
        i += 1;
    }

    // Word-level scan for addresses, codes and capitalised runs.
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut run: Vec<String> = Vec::new();
    for raw in words.iter() {
        let w = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '.' && c != '-');
        if w.is_empty() {
            if !run.is_empty() {
                push(run.join(" "), &mut out, &mut seen);
                run.clear();
            }
            continue;
        }
        if w.contains('@') && w.contains('.') {
            push(w.to_string(), &mut out, &mut seen);
            continue;
        }
        let lower = w.to_lowercase();
        if nonce_like(&lower) {
            push(lower, &mut out, &mut seen);
        }
        let capitalised = w.chars().next().is_some_and(|c| c.is_uppercase())
            && w.chars().skip(1).any(|c| c.is_lowercase());
        if capitalised {
            run.push(w.to_string());
        } else {
            push_run(&run, &mut out, &mut seen);
            run.clear();
        }
    }
    push_run(&run, &mut out, &mut seen);
    let _ = words.len();
    out
}

/// Thread-safe per-`user_id` entity graph.
#[derive(Default)]
pub struct MemoryGraph {
    users: RwLock<HashMap<String, UserGraph>>,
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert(&self, user_id: &str, pair: StoredPair) {
        let mut g = self.users.write().expect("memory graph poisoned");
        g.entry(user_id.to_string()).or_default().upsert(pair);
    }

    pub fn is_empty(&self, user_id: &str) -> bool {
        self.users
            .read()
            .expect("memory graph poisoned")
            .get(user_id)
            .is_none_or(|u| u.pairs.is_empty())
    }

    /// Top passages for a question, by Personalized PageRank over the graph.
    pub fn rank(&self, user_id: &str, query: &str, k: usize) -> Vec<Ranked> {
        self.users
            .read()
            .expect("memory graph poisoned")
            .get(user_id)
            .map(|u| u.rank(query, k))
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(id: &str, ord: usize, prompt: &str, response: &str) -> StoredPair {
        StoredPair {
            pair_id: id.into(),
            session_id: "s".into(),
            timestamp: None,
            prompt: prompt.into(),
            response: response.into(),
            ordinal: ord,
        }
    }

    #[test]
    fn extracts_names_addresses_and_quoted_spans() {
        let e = extract_entities("When I say \"kestrel flight rollout\" I mean Project Kestrel for Faircroft Collective");
        assert!(e.iter().any(|x| x == "kestrel flight rollout"), "{e:?}");
        assert!(e.iter().any(|x| x.contains("faircroft")), "{e:?}");
        let e2 = extract_entities("the new work email is michael.lackey@ravenwyn.com.");
        assert!(e2.iter().any(|x| x.starts_with("michael.lackey@")), "{e2:?}");
    }

    #[test]
    fn one_pass_reaches_the_end_of_a_four_link_chain() {
        let g = MemoryGraph::new();
        for i in 0..80 {
            g.upsert("u", pair(&format!("f{i}"), i, "routine note about the weekly account review", "noted"));
        }
        g.upsert("u", pair("ev0", 80,
            "When I say \"kestrel flight rollout\" I mean the work for Faircroft Collective, owner Michael Lackey", "understood"));
        g.upsert("u", pair("ev1", 81, "Remember: Michael Lackey is my cousin, everyone calls them Micky.", "noted"));
        g.upsert("u", pair("ev2", 82, "Michael Lackey used to work at Penford Partners, now at Ravenwyn Company.", "noted"));
        g.upsert("u", pair("ev3", 83, "After Micky moved to Ravenwyn Company the new email is michael.lackey@ravenwyn.com", "saved"));

        // The question names the project and the client. It never says
        // Michael, Micky or Ravenwyn.
        let hits = g.rank("u", "For the \"kestrel flight rollout\" to Faircroft Collective, which email should I use now?", 12);
        let ids: Vec<&str> = hits.iter().map(|h| h.pair.pair_id.as_str()).collect();
        for want in ["ev0", "ev1", "ev2", "ev3"] {
            assert!(ids.contains(&want), "PPR must reach {want} in one pass: {ids:?}");
        }
    }

    #[test]
    fn graphs_never_cross_users() {
        let g = MemoryGraph::new();
        g.upsert("alice", pair("a1", 0, "Alice Adams works at Northwind Trading", "ok"));
        g.upsert("bob", pair("b1", 0, "Bob Barker works at Southgate Mills", "ok"));
        let hits = g.rank("bob", "where does Alice Adams work?", 5);
        assert!(hits.is_empty(), "bob's graph must not answer from alice's rows");
    }
}
