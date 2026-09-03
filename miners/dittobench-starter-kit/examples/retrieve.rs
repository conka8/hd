//! Retrieval diagnostic: what does the agent actually put in front of the model?
//!
//! A score tells us an answer was wrong but not whether the evidence was ever
//! present. Approximating the ranking in a separate probe would measure the
//! probe, so this drives the real `seed_haystack` and the real recall stage
//! and reports only what they produced.
//!
//! Development tooling. Nothing here is reachable from the served harness.
//!
//!   cargo run --release --example retrieve -- <exam.json> <family> [budget]

use dittobench_starter_kit::baseline::Baseline;
use dittobench_starter_kit::seed::{Pair as SeedPair, SeedRequest};
use serde_json::Value;

/// Characters of a row's opening used to recognise it inside the composed
/// block. Long enough to be unique, short enough to survive any clip.
const FINGERPRINT_CHARS: usize = 48;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let exam = args.get(1).expect("usage: retrieve <exam.json> <family> [budget]");
    let family = args.get(2).cloned().unwrap_or_default();
    let budget: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);

    let doc: Value = serde_json::from_slice(&std::fs::read(exam)?)?;
    let mut pairs = Vec::new();
    for wave in doc["memory_waves"].as_array().into_iter().flatten() {
        for p in wave["pairs"].as_array().into_iter().flatten() {
            pairs.push(p.clone());
        }
    }
    for tc in doc["tool_cases"].as_array().into_iter().flatten() {
        for p in tc["prerequisite_pairs"].as_array().into_iter().flatten() {
            pairs.push(p.clone());
        }
    }

    let agent = Baseline::from_env().await?;
    let user = "retrieve-probe";
    let seed_pairs: Vec<SeedPair> = pairs
        .iter()
        .map(|p| SeedPair {
            pair_id: p["pair_id"].as_str().unwrap_or_default().to_string(),
            session_id: p["session_id"].as_str().unwrap_or_default().to_string(),
            timestamp: p["timestamp"].as_str().unwrap_or_default().to_string(),
            prompt: p["prompt"].as_str().unwrap_or_default().to_string(),
            response: p["response"].as_str().unwrap_or_default().to_string(),
        })
        .collect();
    agent
        .seed_haystack(SeedRequest {
            user_id: Some(user.to_string()),
            wave: 0,
            pairs: seed_pairs,
            subjects: vec![],
            links: vec![],
        })
        .await?;

    // Ground truth for "was the evidence reachable" is the set of rows that
    // literally contain the expected answer. Money answers are stored in
    // minor units, so the decimal rendering is checked too.
    let mut tally: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    // family -> (in_corpus, retrieved, in_context, truncated, complete_context_cases)
    let mut vis: std::collections::BTreeMap<String, (usize, usize, usize, usize, usize)> =
        Default::default();
    let mut block_chars: Vec<usize> = Vec::new();
    // family -> (cases, cluster rows needed, cluster rows present, complete cases)
    let mut comp: std::collections::BTreeMap<String, (usize, usize, usize, usize)> =
        Default::default();
    for case in doc["memory_cases"].as_array().into_iter().flatten() {
        let fam = case["question_type"].as_str().unwrap_or_default();
        if !family.is_empty() && fam != family {
            continue;
        }
        let question = case["question"].as_str().unwrap_or_default();
        let expected = case["expected_answer"].to_string().trim_matches('"').to_string();
        let mut forms = vec![expected.clone()];
        if case["answer_kind"].as_str() == Some("money") {
            if let Ok(cents) = expected.parse::<i64>() {
                forms.push(format!("{}.{:02}", cents / 100, (cents % 100).abs()));
            }
        }
        let present: Vec<&Value> = pairs
            .iter()
            .filter(|p| {
                let hay = format!("{} {}", p["prompt"], p["response"]).to_lowercase();
                forms.iter().any(|f| f.len() > 2 && hay.contains(&f.to_lowercase()))
            })
            .collect();
        if present.is_empty() {
            continue; // computed answer: literal presence says nothing
        }
        let slot = tally.entry(fam.to_string()).or_insert((0, 0));
        slot.0 += 1;
        let block = agent.debug_recall(user, question, budget).unwrap_or_default();
        block_chars.push(block.chars().count());
        let low = block.to_lowercase();

        // A supporting row counts as retrieved if its opening survives into the
        // block, and as visible only if the answer itself does. Separating the
        // two is the whole point: a row can be retrieved perfectly and still
        // have the answer clipped off, which reads downstream as a reasoning
        // failure and is not one.
        // Complete support: every row in the answer row's session cluster
        // except the disposable "-tool" receipt the generator plants as a
        // decoy. For a contact question that is the alias binding, the
        // biography, the superseded address and the current one. The answer
        // row alone is not enough to answer from, which is the whole point of
        // measuring this separately.
        let answer_sess = present
            .first()
            .and_then(|p| p["session_id"].as_str())
            .unwrap_or_default()
            .to_string();
        let root = answer_sess
            .rfind('-')
            .map(|i| &answer_sess[..i])
            .unwrap_or(&answer_sess)
            .to_string();
        let cluster: Vec<&Value> = pairs
            .iter()
            .filter(|p| {
                let sid = p["session_id"].as_str().unwrap_or_default();
                !root.is_empty() && sid.starts_with(&root) && !sid.ends_with("-tool")
            })
            .collect();
        let mut cluster_in_context = 0usize;
        for p in &cluster {
            let head: String = p["prompt"].as_str().unwrap_or_default().chars()
                .take(FINGERPRINT_CHARS).collect::<String>().to_lowercase();
            if head.len() > 12 && low.contains(&head) {
                cluster_in_context += 1;
            }
        }
        let cluster_complete = !cluster.is_empty() && cluster_in_context == cluster.len();
        if std::env::var("PROBE_CLUSTER").is_ok() && !cluster_complete {
            let mut missing = Vec::new();
            for p in &cluster {
                let head: String = p["prompt"].as_str().unwrap_or_default().chars()
                    .take(FINGERPRINT_CHARS).collect::<String>().to_lowercase();
                if !(head.len() > 12 && low.contains(&head)) {
                    missing.push(format!(
                        "{} :: {}",
                        p["session_id"].as_str().unwrap_or_default(),
                        &p["prompt"].as_str().unwrap_or_default()
                            [..p["prompt"].as_str().unwrap_or_default().len().min(90)]
                    ));
                }
            }
            println!("MISSING-CLUSTER [{fam}] {} of {} absent:", missing.len(), cluster.len());
            for m in missing { println!("    {m}"); }
        }
        let cc = comp.entry(fam.to_string()).or_insert((0, 0, 0, 0));
        cc.0 += 1;
        cc.1 += cluster.len();
        cc.2 += cluster_in_context;
        if cluster_complete { cc.3 += 1; }

        let (mut retrieved, mut visible) = (0usize, 0usize);
        for p in &present {
            let head: String = p["prompt"]
                .as_str()
                .unwrap_or_default()
                .chars()
                .take(FINGERPRINT_CHARS)
                .collect::<String>()
                .to_lowercase();
            if head.len() > 12 && low.contains(&head) {
                retrieved += 1;
                if forms
                    .iter()
                    .any(|f| f.len() > 2 && low.contains(&f.to_lowercase()))
                {
                    visible += 1;
                }
            }
        }
        let found = forms.iter().any(|f| f.len() > 2 && low.contains(&f.to_lowercase()));
        if found {
            slot.1 += 1;
        }
        let truncated = retrieved.saturating_sub(visible);
        vis.entry(fam.to_string()).or_insert((0, 0, 0, 0, 0)).0 += present.len();
        {
            let v = vis.get_mut(fam).unwrap();
            v.1 += retrieved;
            v.2 += visible;
            v.3 += truncated;
            if retrieved > 0 && retrieved == present.len() && truncated == 0 {
                v.4 += 1;
            }
        }
        if std::env::var("PROBE_JSON").is_ok() {
            println!(
                "{{\"family\":\"{fam}\",\"support_in_corpus\":{},\"support_retrieved\":{retrieved},\"support_in_context\":{visible},\"support_truncated\":{truncated},\"complete_context\":{}}}",
                present.len(),
                retrieved > 0 && retrieved == present.len() && truncated == 0
            );
        }
        if !found && std::env::var("PROBE_VERBOSE").is_ok() {
            println!("MISS [{fam}] expected={expected} retrieved={retrieved} truncated={truncated}");
            for p in present.iter().take(2) {
                let row = format!("{} || {}", p["prompt"], p["response"]);
                println!("  EVIDENCE(sess={}): {}", p["session_id"], &row[..row.len().min(200)]);
            }
        }
    }

    println!("\n{:38} {:>6} {:>8}", "family", "n", "recall");
    let (mut tn, mut th) = (0usize, 0usize);
    for (fam, (n, h)) in &tally {
        tn += n; th += h;
        println!("{fam:38} {n:6} {:7.1}%", 100.0 * *h as f64 / *n as f64);
    }
    if tn > 0 {
        println!("{:38} {tn:6} {:7.1}%   <- literal-evidence recall @{budget}", "TOTAL", 100.0 * th as f64 / tn as f64);
    }
    println!(
        "\n{:38} {:>9} {:>10} {:>11} {:>10} {:>9}",
        "family", "in_corpus", "retrieved", "in_context", "truncated", "complete"
    );
    let (mut c, mut r, mut i, mut t, mut k) = (0, 0, 0, 0, 0);
    for (fam, (ic, rt, ix, tr, cc)) in &vis {
        c += ic; r += rt; i += ix; t += tr; k += cc;
        println!("{fam:38} {ic:9} {rt:10} {ix:11} {tr:10} {cc:9}");
    }
    println!("{:38} {c:9} {r:10} {i:11} {t:10} {k:9}", "TOTAL");
    println!(
        "\n{:38} {:>6} {:>10} {:>10} {:>18}",
        "family", "cases", "need/case", "have/case", "COMPLETE_SUPPORT"
    );
    for (fam, (n, need, have, done)) in &comp {
        println!(
            "{fam:38} {n:6} {:10.1} {:10.1} {:17.1}%",
            *need as f64 / *n as f64,
            *have as f64 / *n as f64,
            100.0 * *done as f64 / *n as f64
        );
    }
    block_chars.sort_unstable();
    if !block_chars.is_empty() {
        let q = |p: usize| block_chars[(block_chars.len() * p / 100).min(block_chars.len() - 1)];
        println!(
            "\n  recall-context characters: p50={} p95={} p99={} mean={}",
            q(50), q(95), q(99),
            block_chars.iter().sum::<usize>() / block_chars.len()
        );
    }
    if r > 0 {
        println!(
            "  retrieved-but-truncated: {}/{} = {:.1}% of retrieved support never reached the model",
            t, r, 100.0 * t as f64 / r as f64
        );
    }
    Ok(())
}
