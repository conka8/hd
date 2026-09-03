//! Configuration-tool affordance, derived from the live catalog schema.
//!
//! Most tools in a catalog act on the world: send a message, create an event,
//! search the web. A minority act on *the assistant itself*: how much it
//! deliberates, which interface theme it uses, which tools it may reach for.
//!
//! Those two groups are addressed by users in very different ways, and the
//! second is easy to miss. A request to act on the world names its object
//! ("email Priya", "put this on Friday"). A request to reconfigure the
//! assistant often names no object at all and simply describes future conduct:
//! "take your time and reason it through from now on". Measured on the live
//! harness, an explicit "set reasoning effort to high" routed correctly while
//! four behavioural phrasings of the same intent produced no tool call at all.
//! The model was offered the tool every time. It did not recognise that a
//! statement about how it should behave is something a tool can perform.
//!
//! This module closes that gap without knowing anything about any particular
//! tool. It reads the catalog it is handed, decides structurally which entries
//! reconfigure the assistant, recovers each one's permitted values from its own
//! parameter description, and states that affordance compactly. A catalog with
//! no such tools produces nothing; a catalog with new ones picks them up for
//! free.

use crate::protocol::ToolDefWire;

/// Verbs that, at the head of a tool name, denote changing a setting rather
/// than acting on the world. `send_`, `create_`, `search_` and friends all
/// take an external object; these do not.
const CONFIG_VERBS: &[&str] = &[
    "set_", "update_", "enable_", "disable_", "configure_", "toggle_", "switch_",
];

/// A catalog entry that reconfigures the assistant.
#[derive(Debug, Clone)]
pub struct ConfigTool {
    pub name: String,
    /// The single required parameter that carries the new setting.
    pub param: String,
    /// Values the parameter's own description advertises, if it enumerates any.
    pub allowed: Vec<String>,
}

/// Recovers an enumerated value list from a parameter description.
///
/// Catalogs commonly describe a closed set in prose rather than as a JSON
/// enum: "low, medium, or high", "light or dark". This reads that list back
/// out. It deliberately refuses anything that does not look like a short
/// closed set, so a free-text parameter is never mistaken for an enumeration.
pub fn allowed_values(description: &str) -> Vec<String> {
    let cleaned = description.replace(" or ", ", ").replace(" and ", ", ");
    let parts: Vec<String> = cleaned
        .split(',')
        .map(|p| p.trim().trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|p| !p.is_empty())
        .collect();
    // A genuine enumeration is a handful of short single words. Anything
    // longer is prose that happened to contain a comma.
    let plausible = parts.len() >= 2
        && parts.len() <= 6
        && parts.iter().all(|p| p.len() <= 16 && !p.contains(' '));
    if plausible { parts } else { Vec::new() }
}

/// Selects the catalog entries that reconfigure the assistant.
///
/// Structural test, not a name list: a configuration verb at the head of the
/// name, and exactly one required parameter carrying the new value. A tool
/// that mutates the world takes an object to mutate, so it fails the second
/// condition or does not begin with one of these verbs.
pub fn detect(tools: &[ToolDefWire]) -> Vec<ConfigTool> {
    let mut out = Vec::new();
    for t in tools {
        let lower = t.name.to_lowercase();
        if !CONFIG_VERBS.iter().any(|v| lower.starts_with(v)) {
            continue;
        }
        let required = t.parameters.get("required").and_then(|r| r.as_array());
        let Some(required) = required else { continue };
        if required.len() != 1 {
            continue;
        }
        let Some(param) = required[0].as_str() else { continue };
        let desc = t
            .parameters
            .get("properties")
            .and_then(|p| p.get(param))
            .and_then(|p| p.get("description"))
            .and_then(|d| d.as_str())
            .unwrap_or_default();
        out.push(ConfigTool {
            name: t.name.clone(),
            param: param.to_string(),
            allowed: allowed_values(desc),
        });
    }
    out
}

/// Renders the affordance for the prompt, or nothing when the catalog offers
/// no configuration tools.
///
/// States only what the schema says. The mapping from a user's wording to a
/// permitted value is left to the model, which is the part that genuinely
/// needs language understanding; naming the tools and their value sets is the
/// part it was missing.
pub fn hint(tools: &[ConfigTool]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let mut s = String::from(
        "\nSome tools here change how YOU operate rather than acting on the world. \
Any stated preference about your own manner of working is one of these, whether it \
asks for MORE of something or LESS: deliberate harder or go faster, be thorough or \
be brief, change how you present yourself, change what you may use. Phrasings like \
'from now on', 'going forward', or a plain preference with no object at all are \
still setting changes. Perform it by calling the matching tool exactly once. \
Agreeing in words without calling it changes nothing. Pick the value from that \
tool's own list that best matches the direction asked for:\n",
    );
    for t in tools {
        if t.allowed.is_empty() {
            s.push_str(&format!("  {}({}: <value>)\n", t.name, t.param));
        } else {
            s.push_str(&format!(
                "  {}({}: {})\n",
                t.name,
                t.param,
                t.allowed.join(" | ")
            ));
        }
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn wire(name: &str, param: &str, desc: &str) -> ToolDefWire {
        ToolDefWire {
            name: name.to_string(),
            description: "d".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { param: {"type": "string", "description": desc} },
                "required": [param]
            }),
        }
    }

    #[test]
    fn reads_enumerations_out_of_parameter_prose() {
        assert_eq!(allowed_values("low, medium, or high"), vec!["low", "medium", "high"]);
        assert_eq!(allowed_values("light or dark"), vec!["light", "dark"]);
    }

    #[test]
    fn refuses_to_treat_free_text_as_an_enumeration() {
        for prose in [
            "the search query to run",
            "a natural language description of the task, including any details",
            "",
        ] {
            assert!(allowed_values(prose).is_empty(), "should not enumerate {prose:?}");
        }
    }

    #[test]
    fn selects_configuration_tools_and_ignores_world_actions() {
        let tools = vec![
            wire("set_reasoning_effort", "effort", "low, medium, or high"),
            wire("set_theme", "theme", "light or dark"),
            wire("search_web", "query", "the search query"),
            wire("execute_agent_job", "task", "the task to run"),
        ];
        let found = detect(&tools);
        let names: Vec<&str> = found.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["set_reasoning_effort", "set_theme"]);
        assert_eq!(found[0].allowed, vec!["low", "medium", "high"]);
    }

    #[test]
    fn produces_nothing_for_a_catalog_without_configuration_tools() {
        let tools = vec![wire("search_web", "query", "the search query")];
        assert!(detect(&tools).is_empty());
        assert!(hint(&detect(&tools)).is_none());
    }

    #[test]
    fn hint_names_each_tool_with_its_own_permitted_values() {
        let tools = vec![wire("set_reasoning_effort", "effort", "low, medium, or high")];
        let h = hint(&detect(&tools)).expect("hint");
        assert!(h.contains("set_reasoning_effort(effort: low | medium | high)"), "{h}");
    }
}

/// A resolved configuration change: which tool, and what value.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub tool: String,
    pub param: String,
    pub value: String,
}

/// Parses the constrained reply of the configuration adjudicator.
///
/// The adjudicator is asked for one line, either `NONE` or `<tool> <value>`.
/// Anything it returns is checked back against the catalog: the tool must be
/// one we offered, and where that tool advertises a closed value set the value
/// must be in it. A reply that fails either check is discarded rather than
/// coerced, so a confused adjudicator degrades to "no change" instead of
/// inventing a setting.
pub fn parse_decision(reply: &str, tools: &[ConfigTool]) -> Option<Decision> {
    let line = reply
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or_default()
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != ' ');
    if line.is_empty() || line.eq_ignore_ascii_case("none") {
        return None;
    }
    let mut parts = line.split_whitespace();
    let name = parts.next()?.to_lowercase();
    let value = parts.next()?.to_lowercase();
    let tool = tools.iter().find(|t| t.name.eq_ignore_ascii_case(&name))?;
    if !tool.allowed.is_empty() && !tool.allowed.iter().any(|a| a == &value) {
        return None;
    }
    Some(Decision { tool: tool.name.clone(), param: tool.param.clone(), value })
}

/// The adjudicator prompt: one narrow decision, with the catalog in view.
///
/// Deliberately not a general assistant turn. It asks a single closed
/// question, which is the kind of judgement a language model is reliable at,
/// and it is only ever reached when the ordinary path already declined to act.
pub fn decision_prompt(request: &str, tools: &[ConfigTool]) -> String {
    let mut s = String::from(
        "Decide whether this user message asks to change one of the assistant's own \
settings. A stated preference about how the assistant should work counts, in \
either direction, however informally it is phrased. A request about anything in \
the outside world does not.\n\nSettings available:\n",
    );
    for t in tools {
        if t.allowed.is_empty() {
            s.push_str(&format!("  {} <value>\n", t.name));
        } else {
            s.push_str(&format!("  {} [{}]\n", t.name, t.allowed.join("|")));
        }
    }
    s.push_str("\nUser message:\n");
    s.push_str(request);
    s.push_str(
        "\n\nReply with exactly one line and nothing else: either NONE, or the tool \
name followed by the value.\n",
    );
    s
}

#[cfg(test)]
mod decision_tests {
    use super::*;


    fn cfg() -> Vec<ConfigTool> {
        vec![
            ConfigTool { name: "set_reasoning_effort".into(), param: "effort".into(),
                         allowed: vec!["low".into(), "medium".into(), "high".into()] },
            ConfigTool { name: "set_theme".into(), param: "theme".into(),
                         allowed: vec!["light".into(), "dark".into()] },
        ]
    }

    #[test]
    fn accepts_a_well_formed_decision() {
        let d = parse_decision("set_reasoning_effort low", &cfg()).expect("decision");
        assert_eq!(d, Decision { tool: "set_reasoning_effort".into(),
                                 param: "effort".into(), value: "low".into() });
    }

    #[test]
    fn none_means_leave_the_answer_alone() {
        for reply in ["NONE", "none", "  none  ", ""] {
            assert!(parse_decision(reply, &cfg()).is_none(), "{reply:?}");
        }
    }

    #[test]
    fn rejects_values_outside_the_advertised_set() {
        // Must not coerce "maximum" to "high": an unrecognised value means the
        // adjudicator misunderstood, and inventing a setting is worse than none.
        assert!(parse_decision("set_reasoning_effort maximum", &cfg()).is_none());
        assert!(parse_decision("set_reasoning_effort", &cfg()).is_none());
    }

    #[test]
    fn rejects_tools_that_were_never_offered() {
        assert!(parse_decision("set_volume high", &cfg()).is_none());
        assert!(parse_decision("search_web cats", &cfg()).is_none());
    }

    #[test]
    fn prompt_lists_every_setting_with_its_values() {
        let p = decision_prompt("go faster", &cfg());
        assert!(p.contains("set_reasoning_effort [low|medium|high]"), "{p}");
        assert!(p.contains("set_theme [light|dark]"));
        assert!(p.contains("go faster"));
    }
}
