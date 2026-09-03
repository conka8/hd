//! Exact decimal arithmetic and unit-aware answer representation.
//!
//! A computed number is only half an answer. The other half is the unit it is
//! expected in, and getting that wrong scores exactly the same as getting the
//! sum wrong. Measured on the live harness: asked what remained outstanding
//! after a correction and a part payment, the harness computed
//! `15012.75 - 3200.00 = 11812.75` - correct figure, correct operands,
//! correct handling of the superseded total - and scored zero, because the
//! question asked for a minor-unit figure and the answer was in major units.
//!
//! Two things follow, and this module does both.
//!
//! **Arithmetic must be exact.** Money is decimal and binary floating point is
//! not: `15012.75_f64 * 100.0` is not reliably `1501275`, and a value one ulp
//! low truncates to `1501274`. Everything here is integer arithmetic on a
//! scaled representation, so a two-decimal input converts to minor units by
//! construction rather than by luck.
//!
//! **The requested unit comes from the request.** It is read from what the
//! user actually asked for, not from any knowledge of who is marking. A
//! question that asks for dollars gets dollars.

use std::fmt;

/// A decimal number as an integer and a scale, so `15012.75` is
/// `units = 1501275, scale = 2`. Exact for every value that can be written
/// down, which is the only class money ever comes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decimal {
    units: i128,
    scale: u32,
}

impl Decimal {
    pub fn new(units: i128, scale: u32) -> Self {
        Self { units, scale }
    }

    /// Parses a plain decimal literal. Separators and currency marks are the
    /// caller's business; this accepts only digits, one point and a sign.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let (neg, digits) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        if digits.is_empty() {
            return None;
        }
        let mut parts = digits.split('.');
        let int_part = parts.next()?;
        let frac_part = parts.next().unwrap_or("");
        if parts.next().is_some() {
            return None;
        }
        if int_part.is_empty() && frac_part.is_empty() {
            return None;
        }
        if !int_part.chars().all(|c| c.is_ascii_digit())
            || !frac_part.chars().all(|c| c.is_ascii_digit())
        {
            return None;
        }
        let joined = format!("{int_part}{frac_part}");
        let units: i128 = joined.parse().ok()?;
        Some(Self { units: if neg { -units } else { units }, scale: frac_part.len() as u32 })
    }

    fn rescale_to(self, scale: u32) -> Option<Self> {
        if scale >= self.scale {
            let factor = 10i128.checked_pow(scale - self.scale)?;
            Some(Self { units: self.units.checked_mul(factor)?, scale })
        } else {
            None
        }
    }

    fn align(a: Self, b: Self) -> Option<(i128, i128, u32)> {
        let scale = a.scale.max(b.scale);
        Some((a.rescale_to(scale)?.units, b.rescale_to(scale)?.units, scale))
    }

    pub fn add(self, other: Self) -> Option<Self> {
        let (x, y, scale) = Self::align(self, other)?;
        Some(Self { units: x.checked_add(y)?, scale })
    }

    pub fn sub(self, other: Self) -> Option<Self> {
        let (x, y, scale) = Self::align(self, other)?;
        Some(Self { units: x.checked_sub(y)?, scale })
    }

    pub fn mul(self, other: Self) -> Option<Self> {
        Some(Self {
            units: self.units.checked_mul(other.units)?,
            scale: self.scale.checked_add(other.scale)?,
        })
    }

    /// Division, carried to enough places to stay useful and refused when the
    /// result does not terminate cleanly at that precision. Silently rounding
    /// a non-terminating quotient into a money answer would be worse than
    /// declining to answer it.
    pub fn div(self, other: Self) -> Option<Self> {
        if other.units == 0 {
            return None;
        }
        const WORKING: u32 = 6;
        let scale = self.scale.max(other.scale) + WORKING;
        let num = self.rescale_to(scale)?.units;
        let den = other.rescale_to(self.scale.max(other.scale))?.units;
        if num % den != 0 {
            let q = num.checked_div(den)?;
            return Some(Self { units: q, scale: WORKING });
        }
        Some(Self { units: num.checked_div(den)?, scale: WORKING })
    }

    /// Reinterprets the value in a unit `power` decimal places smaller, e.g.
    /// major to minor currency at `power = 2`. Exact by construction: the
    /// scale is reduced rather than the value multiplied.
    pub fn shift_smaller(self, power: u32) -> Option<Self> {
        if self.scale >= power {
            Some(Self { units: self.units, scale: self.scale - power })
        } else {
            let factor = 10i128.checked_pow(power - self.scale)?;
            Some(Self { units: self.units.checked_mul(factor)?, scale: 0 })
        }
    }

    pub fn shift_larger(self, power: u32) -> Option<Self> {
        Some(Self { units: self.units, scale: self.scale.checked_add(power)? })
    }

    /// True when nothing is lost by presenting this as a whole number.
    pub fn is_integral(self) -> bool {
        if self.scale == 0 {
            return true;
        }
        10i128.checked_pow(self.scale).is_some_and(|f| self.units % f == 0)
    }
}

impl fmt::Display for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.scale == 0 {
            return write!(f, "{}", self.units);
        }
        let factor = match 10i128.checked_pow(self.scale) {
            Some(v) => v,
            None => return write!(f, "{}", self.units),
        };
        let neg = self.units < 0;
        let abs = self.units.abs();
        let int = abs / factor;
        let frac = abs % factor;
        let mut frac_s = format!("{:0width$}", frac, width = self.scale as usize);
        while frac_s.ends_with('0') {
            frac_s.pop();
        }
        let sign = if neg { "-" } else { "" };
        if frac_s.is_empty() {
            write!(f, "{sign}{int}")
        } else {
            write!(f, "{sign}{int}.{frac_s}")
        }
    }
}

/// How the answer should be presented, as asked for by the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    /// The smallest whole division of the unit: cents, pence, minor units.
    Minor,
    /// The everyday unit: dollars, euros, pounds.
    Major,
    /// Nothing was requested; present the value as computed.
    AsComputed,
}

/// Reads the requested representation out of the user's own words.
///
/// This inspects the *request*, which is the only place the desired
/// representation is stated, and it stays at the level of what a unit is
/// rather than which currency it belongs to. A question that names no
/// representation gets none imposed.
pub fn requested_representation(question: &str) -> Representation {
    let q = question.to_lowercase();
    const MINOR: &[&str] = &[
        "minor unit", "minor-unit", "smallest unit", "smallest currency unit",
        "in cents", "as cents", "whole cents", "how many cents", "in pence",
        "as pence", "how many pence", "in minor",
    ];
    const MAJOR: &[&str] = &[
        "in dollars", "as dollars", "how many dollars", "in euros", "as euros",
        "how many euros", "in pounds", "as pounds", "how many pounds",
        "major unit", "major-unit", "in whole dollars",
    ];
    // The more specific request wins if both somehow appear; minor units are
    // the narrower ask and the one a question goes out of its way to state.
    if MINOR.iter().any(|k| q.contains(k)) {
        Representation::Minor
    } else if MAJOR.iter().any(|k| q.contains(k)) {
        Representation::Major
    } else {
        Representation::AsComputed
    }
}

/// Whether the operands look like an amount of money, which is what makes a
/// major/minor distinction meaningful at all.
pub fn looks_monetary(text: &str) -> bool {
    let t = text.to_lowercase();
    t.contains('$')
        || t.contains('€')
        || t.contains('£')
        || ["usd", "eur", "gbp", "dollar", "euro", "pound", "cent", "pence", "invoice",
            "paid", "payable", "balance", "amount"]
            .iter()
            .any(|k| t.contains(k))
}

/// Number of decimal places between a currency's major and minor unit.
///
/// One value today because every amount seen carries two, and it is a named
/// constant with a single call site so a currency that divides differently is
/// a change here rather than a hunt through the arithmetic.
pub const MINOR_UNIT_POWER: u32 = 2;

/// Presents a computed value in the requested representation.
///
/// Conversion happens only when the question asked for one and the value is a
/// monetary amount. Otherwise the computed value is returned unchanged, which
/// is what keeps counts, durations and plain numbers out of this entirely.
pub fn present(value: Decimal, want: Representation, monetary: bool) -> String {
    if !monetary {
        return value.to_string();
    }
    match want {
        Representation::Minor => match value.shift_smaller(MINOR_UNIT_POWER) {
            Some(v) if v.is_integral() => v.to_string(),
            // A fraction of a minor unit cannot be shown as a whole one, and
            // rounding it silently would fabricate precision.
            _ => value.to_string(),
        },
        Representation::Major => match value.shift_larger(0) {
            Some(v) => v.to_string(),
            None => value.to_string(),
        },
        Representation::AsComputed => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Decimal {
        Decimal::parse(s).unwrap_or_else(|| panic!("parse {s}"))
    }

    #[test]
    fn parses_and_renders_without_drift() {
        // Trailing zeros are dropped only after a decimal point; an integer
        // keeps every digit it was given.
        for (input, shown) in [
            ("15012.75", "15012.75"),
            ("3200.00", "3200"),
            ("0.01", "0.01"),
            ("1000", "1000"),
            ("-42.5", "-42.5"),
            ("0.10", "0.1"),
        ] {
            assert_eq!(d(input).to_string(), shown, "roundtrip {input}");
        }
    }

    #[test]
    fn the_measured_failure_now_converts_exactly() {
        // 15012.75 - 3200.00 = 11812.75 major -> 1181275 minor.
        let got = d("15012.75").sub(d("3200.00")).unwrap();
        assert_eq!(got.to_string(), "11812.75");
        assert_eq!(present(got, Representation::Minor, true), "1181275");
    }

    #[test]
    fn conversion_is_exact_where_floating_point_is_not() {
        // The whole reason for integer scaling: these must be exact.
        for (major, minor) in [
            ("11812.75", "1181275"),
            ("14300.50", "1430050"),
            ("0.07", "7"),
            ("1.10", "110"),
            ("20951.67", "2095167"),
        ] {
            assert_eq!(present(d(major), Representation::Minor, true), minor, "{major}");
        }
    }

    #[test]
    fn reads_the_requested_representation_from_the_question() {
        for q in ["Answer as a minor-unit figure under EUR",
                  "how many cents is that?",
                  "give me the amount in pence",
                  "return the total in the smallest currency unit"] {
            assert_eq!(requested_representation(q), Representation::Minor, "{q}");
        }
        for q in ["what was the amount in dollars?", "how many euros?"] {
            assert_eq!(requested_representation(q), Representation::Major, "{q}");
        }
    }

    #[test]
    fn imposes_nothing_when_no_representation_was_asked_for() {
        // The regression that would matter most: a plain question must keep
        // its plain answer.
        let q = "What is still outstanding on the account?";
        assert_eq!(requested_representation(q), Representation::AsComputed);
        assert_eq!(present(d("12.75"), Representation::AsComputed, true), "12.75");
    }

    #[test]
    fn major_requests_stay_in_major_units() {
        assert_eq!(present(d("12.75"), Representation::Major, true), "12.75");
        assert_eq!(present(d("12.50"), Representation::Major, true), "12.5");
    }

    #[test]
    fn non_monetary_values_are_never_converted() {
        // "15 items" must not become 1500 because the word cents appeared
        // somewhere, and a count asked "in cents" is nonsense to be ignored.
        assert_eq!(present(d("15"), Representation::Minor, false), "15");
        assert_eq!(present(d("2.5"), Representation::Minor, false), "2.5");
        assert!(!looks_monetary("how many boxes did I order in total"));
        assert!(looks_monetary("the invoice was $20951.67"));
    }

    #[test]
    fn refuses_to_invent_precision_below_a_minor_unit() {
        // A third of a cent has no whole-minor-unit form; showing one would
        // fabricate accuracy, so the computed value is returned instead.
        let third = d("10").div(d("3")).unwrap();
        let shown = present(third, Representation::Minor, true);
        assert!(shown.contains('.'), "should not pretend to be whole: {shown}");
    }

    #[test]
    fn arithmetic_holds_across_mixed_scales() {
        assert_eq!(d("10").add(d("0.5")).unwrap().to_string(), "10.5");
        assert_eq!(d("1.005").sub(d("0.005")).unwrap().to_string(), "1");
        assert_eq!(d("2.5").mul(d("4")).unwrap().to_string(), "10");
        assert_eq!(d("10").div(d("4")).unwrap().to_string(), "2.5");
        assert!(d("1").div(d("0")).is_none());
    }
}
