//! Semantic execution tracing for offline analysis.
//!
//! Off by default. With `--features analysis_trace` every scored request emits
//! a JSONL event stream plus a verbatim capture of what the model was shown.
//!
//! # Why it is neutral by construction
//!
//! [`CaseTrace`] exists in **both** builds and every signature that carries one
//! is identical either way. Without the feature the struct holds nothing, its
//! methods have empty bodies, and the value closures passed to [`CaseTrace::ev`]
//! are never called, so no `json!` is ever built. There is no `#[cfg]` on any
//! function signature, which is what usually makes instrumented and clean
//! builds drift apart.
//!
//! Nothing here reads evaluation data. Scores, expected answers and family
//! labels are joined afterwards by the analysis harness, on its own side of the
//! boundary.

use serde_json::Value;

/// Where an event happened. `line!()` at the call site keeps this honest.
#[derive(Clone, Copy)]
pub struct Src {
    pub file: &'static str,
    pub function: &'static str,
    pub line: u32,
}

#[macro_export]
macro_rules! src_here {
    ($func:expr) => {
        $crate::trace::Src { file: file!(), function: $func, line: line!() }
    };
}

#[cfg(feature = "analysis_trace")]
mod imp {
    use super::{Src, Value};
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    /// Redaction is centralised deliberately: a per-field discipline is one
    /// forgotten field away from writing a key into a file meant for sharing.
    const SECRET_KEYS: [&str; 8] = [
        "api_key", "apikey", "authorization", "auth", "token", "secret",
        "password", "openrouter_api_key",
    ];

    pub fn redact(v: &Value) -> Value {
        match v {
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, val)| {
                        let lk = k.to_ascii_lowercase();
                        if SECRET_KEYS.iter().any(|s| lk.contains(s)) {
                            (k.clone(), Value::String("[redacted]".into()))
                        } else {
                            (k.clone(), redact(val))
                        }
                    })
                    .collect(),
            ),
            Value::Array(a) => Value::Array(a.iter().map(redact).collect()),
            Value::String(s) => {
                // Catch a key pasted into free text, not just into a named field.
                if s.starts_with("sk-or-") || s.starts_with("Bearer ") {
                    Value::String("[redacted]".into())
                } else {
                    v.clone()
                }
            }
            _ => v.clone(),
        }
    }

    /// One writer per process, not per case.
    ///
    /// Cases run concurrently, and a separate `File` handle per case interleaves
    /// partial lines into the shared JSONL even in append mode. The validator
    /// caught exactly that: 92 corrupt lines out of 1292. A single handle behind
    /// one mutex serialises whole lines.
    static SINK: std::sync::OnceLock<Mutex<Option<std::fs::File>>> = std::sync::OnceLock::new();

    fn sink_for(path: &std::path::Path) -> &'static Mutex<Option<std::fs::File>> {
        SINK.get_or_init(|| {
            Mutex::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .ok(),
            )
        })
    }

    pub struct CaseTrace {
        run_id: String,
        case_id: String,
        seq: AtomicU64,
        started: std::time::Instant,
        path: std::path::PathBuf,
        dir: std::path::PathBuf,
    }

    impl CaseTrace {
        pub fn new(run_id: &str, case_id: &str) -> CaseTrace {
            let dir = std::path::PathBuf::from(
                std::env::var("DITTOBENCH_TRACE_DIR").unwrap_or_else(|_| "traces".into()),
            );
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("{run_id}.jsonl"));
            CaseTrace {
                run_id: run_id.to_string(),
                case_id: case_id.to_string(),
                seq: AtomicU64::new(0),
                started: std::time::Instant::now(),
                path,
                dir,
            }
        }

        pub fn enabled(&self) -> bool {
            true
        }

        pub fn ev<F: FnOnce() -> Value>(&self, stage: &str, event: &str, src: Src, values: F) {
            let seq = self.seq.fetch_add(1, Ordering::SeqCst);
            let line = serde_json::json!({
                "schema_version": 1,
                "run_id": self.run_id,
                "case_id": self.case_id,
                "seq": seq,
                "elapsed_us": self.started.elapsed().as_micros() as u64,
                "stage": stage,
                "event": event,
                "source": {"file": src.file, "function": src.function, "line": src.line},
                "values": redact(&values()),
            });
            if let Ok(mut guard) = sink_for(&self.path).lock() {
                if let Some(f) = guard.as_mut() {
                    // One write call per line, under the lock, so concurrent
                    // cases cannot split each other's JSON.
                    let mut buf = line.to_string().into_bytes();
                    buf.push(b'\n');
                    let _ = f.write_all(&buf);
                }
            }
        }

        /// The verbatim model context, written beside the event stream because
        /// it is large and wanted whole rather than in fragments.
        pub fn capture_context(&self, label: &str, payload: Value) {
            let path = self
                .dir
                .join(format!("{}.{}.{}.context.json", self.run_id, self.case_id, label));
            if let Ok(s) = serde_json::to_string_pretty(&redact(&payload)) {
                let _ = std::fs::write(path, s);
            }
        }
    }
}

#[cfg(not(feature = "analysis_trace"))]
mod imp {
    use super::{Src, Value};

    /// The disabled tracer. Same shape, no behaviour, no allocation.
    pub struct CaseTrace;

    impl CaseTrace {
        #[inline(always)]
        pub fn new(_run_id: &str, _case_id: &str) -> CaseTrace {
            CaseTrace
        }
        #[inline(always)]
        pub fn enabled(&self) -> bool {
            false
        }
        /// `values` is never called, so the `json!` at the call site costs
        /// nothing in a normal build.
        #[inline(always)]
        pub fn ev<F: FnOnce() -> Value>(&self, _stage: &str, _event: &str, _src: Src, _values: F) {}
        #[inline(always)]
        pub fn capture_context(&self, _label: &str, _payload: Value) {}
    }
}

pub use imp::CaseTrace;
