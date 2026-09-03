//! Whether a proposed tool call is actually authorised by the current request.
//!
//! Selecting a plausible tool and being entitled to run it are different
//! questions, and the planner only answers the first. An agent holding a
//! toolbox will reach for it whenever a tool *matches the topic*, which is not
//! the same as the user having asked for the thing the tool does. Asking "which
//! accent colour did I pick?" matches a colour-setting tool perfectly and
//! authorises nothing.
//!
//! So this sits between selection and execution and answers the second
//! question on its own terms. It is deliberately not a model call: the same
//! model that proposed an unnecessary action is a poor judge of whether the
//! action was necessary, and it has already ignored being told so in the
//! prompt.
//!
//! The rule it enforces is a general one about authority:
//!
//!   Recalled facts may supply an action's ARGUMENTS. They cannot supply the
//!   AUTHORITY to take it. That has to come from the current request.
//!
//! "I prefer dark mode" plus "what theme do I prefer?" yields an answer.
//! "I prefer dark mode" plus "switch to the theme I prefer" yields an action.
//! The stored fact is identical; only the current request differs.

use serde_json::Value;

/// What running a tool does to the world.
///
/// Derived from the tool's own schema and description, because that is the
/// only description of a tool that is available in general. Names are a hint,
/// never the sole signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// Returns information and changes nothing. Safe to run whenever the
    /// answer is not already known.
    ReadOnly,
    /// Alters durable state or a setting.
    StateChange,
    /// Runs supplied work and returns its result. No durable state, but the
    /// user has to have supplied the work.
    Execution,
    /// Creates something addressable.
    Create,
    /// Removes something.
    Delete,
    /// Not confidently classifiable; treated as read-only, since the cost of
    /// wrongly suppressing a read is higher than the cost of allowing one.
    Unknown,
}

impl Effect {
    /// Whether running this leaves a trace the user did not ask for.
    pub fn has_side_effect(self) -> bool {
        matches!(self, Effect::StateChange | Effect::Create | Effect::Delete)
    }
}

/// Classifies a tool by what it does, from its name and description.
pub fn classify(name: &str, description: &str) -> Effect {
    let n = name.to_ascii_lowercase();
    let d = description.to_ascii_lowercase();

    // Execution first: a code runner often describes itself in terms that
    // would otherwise read as retrieval ("returns the result").
    const EXEC: [&str; 5] = ["run_code", "execute_code", "eval", "interpreter", "sandbox"];
    if EXEC.iter().any(|m| n.contains(m)) || d.contains("execute the code") {
        return Effect::Execution;
    }
    const DELETE: [&str; 3] = ["delete_", "remove_", "clear_"];
    if DELETE.iter().any(|m| n.starts_with(m)) {
        return Effect::Delete;
    }
    const CREATE: [&str; 4] = ["create_", "new_", "add_", "save_"];
    if CREATE.iter().any(|m| n.starts_with(m)) {
        return Effect::Create;
    }
    const SET: [&str; 7] = [
        "set_", "update_", "enable_", "disable_", "configure_", "toggle_", "switch_",
    ];
    if SET.iter().any(|m| n.starts_with(m)) {
        return Effect::StateChange;
    }
    const READ: [&str; 8] = [
        "list_", "get_", "search_", "find_", "fetch_", "read_", "lookup_", "discover_",
    ];
    if READ.iter().any(|m| n.starts_with(m)) {
        return Effect::ReadOnly;
    }
    Effect::Unknown
}

/// Why a call was refused, for the message handed back to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// A computation was proposed over values the request did not supply.
    OperandsNotSupplied,
}

impl Refusal {
    pub fn message(&self) -> &'static str {
        match self {
            Refusal::OperandsNotSupplied => {
                "Not run: this request did not supply values to compute with. The figures \
                 for it are the ones recalled above, so state the arithmetic on the \
                 COMPUTE line and it will be evaluated exactly."
            }
        }
    }
}

/// Numeric literals the user wrote in the current request.
///
/// Digits attached to letters are identifiers rather than quantities and are
/// skipped, the same reading used when indexing figures out of recalled rows.
fn supplied_numbers(request: &str) -> usize {
    let chars: Vec<char> = request.chars().collect();
    let mut count = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len()
            && (chars[i].is_ascii_digit()
                || ((chars[i] == ',' || chars[i] == '.')
                    && i + 1 < chars.len()
                    && chars[i + 1].is_ascii_digit()))
        {
            i += 1;
        }
        let touches_alpha = (start > 0 && chars[start - 1].is_ascii_alphabetic())
            || chars.get(i).is_some_and(|c| c.is_ascii_alphabetic());
        if !touches_alpha {
            count += 1;
        }
    }
    count
}

/// Fewest numbers a request must contain before a computation over them counts
/// as work the user handed over. One number is a mention; two can be combined.
const MIN_SUPPLIED_OPERANDS: usize = 2;

/// Decides whether a proposed call may run.
///
/// Only [`Effect::Execution`] is gated here. Running work the user did not
/// supply is the one case that is unambiguous: if the operands are not in the
/// request, the request is a question about records, and a question about
/// records is answered from records.
///
/// Reads are never suppressed. "List my automations" is phrased as a question
/// and genuinely needs the tool, and wrongly refusing a read costs more than
/// wrongly allowing one. State changes are not gated yet either; that needs a
/// reliable reading of whether the user asked for the change to happen now,
/// which the data does not yet support.
pub fn authorize(request: &str, name: &str, description: &str, _args: &Value) -> Result<(), Refusal> {
    match classify(name, description) {
        Effect::Execution if supplied_numbers(request) < MIN_SUPPLIED_OPERANDS => {
            Err(Refusal::OperandsNotSupplied)
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn allowed(req: &str) -> bool {
        authorize(req, "run_code", "Run a snippet and return its output", &json!({})).is_ok()
    }

    #[test]
    fn computation_is_allowed_when_the_request_supplies_the_values() {
        assert!(allowed(
            "Can you crunch this for me? my last three readings were 312, 277, and 1510. \
             What is the mean, and how far apart are the high and low?"
        ));
        assert!(allowed("just compute 312 times 12 minus 310 right now."));
    }

    #[test]
    fn computation_is_refused_when_the_values_come_from_records() {
        assert!(!allowed(
            "Reconciling my accounts. What is the current balance owed on Gia Callahan's \
             account? Answer with the dollar amount."
        ));
        assert!(!allowed(
            "For \"riverline rollout\", what is still owed once the approved correction \
             and the payment already sent are reconciled?"
        ));
    }

    #[test]
    fn identifiers_are_not_operands() {
        // "AP-C27330FE" and a lone year are not two numbers to combine.
        assert!(!allowed("Reconcile invoice AP-C27330FE for the 2026 engagement."));
    }

    #[test]
    fn reads_are_never_suppressed() {
        assert!(authorize(
            "Which automations do I have?",
            "list_workflows",
            "List the user's saved workflows",
            &json!({})
        )
        .is_ok());
    }

    #[test]
    fn effects_are_read_off_the_catalog() {
        assert_eq!(classify("set_theme", "Set the UI theme"), Effect::StateChange);
        assert_eq!(classify("list_workflows", "List workflows"), Effect::ReadOnly);
        assert_eq!(classify("run_code", "Run a snippet"), Effect::Execution);
        assert_eq!(classify("delete_memory", "Delete a memory"), Effect::Delete);
        assert_eq!(classify("create_image", "Create an image"), Effect::Create);
        assert!(Effect::StateChange.has_side_effect());
        assert!(!Effect::ReadOnly.has_side_effect());
    }
}
