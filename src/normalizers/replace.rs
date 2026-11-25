use std::borrow::Cow;

use fancy_regex::{NoExpand, Regex};
use serde::de::Error as _;
use serde_json::Value;

use super::Error;

#[derive(Clone, Debug)]
enum Pattern {
    Literal(String),
    Regex(Regex),
}

impl Pattern {
    fn from_json(value: Value) -> Result<Self, Error> {
        if let Some(s) = value.as_str() {
            return Ok(Self::Literal(s.to_string()));
        }

        let obj = value.as_object().ok_or_else(|| {
            serde_json::Error::custom("Replace.pattern must be a string or an object")
        })?;

        if let Some(literal) = obj.get("String").and_then(Value::as_str) {
            return Ok(Self::Literal(literal.to_string()));
        }

        if let Some(regex) = obj.get("Regex").and_then(Value::as_str) {
            return Ok(Self::Regex(Regex::new(regex)?));
        }

        Err(serde_json::Error::custom("Replace.pattern object must contain String or Regex").into())
    }
}

/// Replace normalizer.
///
/// Replaces each occurrence of a literal string or regex pattern with
/// `content`.
#[derive(Clone, Debug)]
pub struct Replace {
    pattern: Pattern,
    content: String,
}

impl Replace {
    pub fn from_config(pattern: Value, content: String) -> Result<Self, Error> {
        Ok(Self {
            pattern: Pattern::from_json(pattern)?,
            content,
        })
    }

    pub fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        match &self.pattern {
            Pattern::Literal(needle) => replace_literal(input, needle, &self.content),
            Pattern::Regex(re) => re.replace_all(input, NoExpand(&self.content)),
        }
    }
}

fn replace_literal<'a>(input: &'a str, needle: &str, replacement: &str) -> Cow<'a, str> {
    if needle.is_empty() {
        let mut output = String::new();
        output.push_str(replacement);
        for ch in input.chars() {
            output.push(ch);
            output.push_str(replacement);
        }
        return Cow::Owned(output);
    }

    // A long input with a one-char needle (Gemma's and Llama's " " → "▁") is
    // replaced in chunks on the pool: such a match never straddles a char
    // boundary, so chunks cut at char boundaries are replaced independently.
    let threads = crate::fanout::threads();
    if input.len() >= PARALLEL_MIN && threads > 1 && needle.chars().nth(1).is_none() {
        let chunks = (input.len() / CHUNK_BYTES).min(threads * 4);
        let mut bounds = Vec::with_capacity(chunks + 1);
        bounds.push(0);
        for k in 1..chunks {
            let mut at = k * input.len() / chunks;
            while !input.is_char_boundary(at) {
                at += 1;
            }
            bounds.push(at);
        }
        bounds.push(input.len());
        let parts = crate::fanout::map(chunks, |k| {
            replace_literal_in(&input[bounds[k]..bounds[k + 1]], needle, replacement)
        });
        if parts.iter().all(|p| matches!(p, Cow::Borrowed(_))) {
            return Cow::Borrowed(input);
        }
        let mut output = String::with_capacity(parts.iter().map(|p| p.len()).sum());
        for part in &parts {
            output.push_str(part);
        }
        return Cow::Owned(output);
    }
    replace_literal_in(input, needle, replacement)
}

/// An input of at least this many bytes is replaced in chunks on the pool.
const PARALLEL_MIN: usize = 128 * 1024;
/// About how many bytes each chunk replaced on the pool holds.
const CHUNK_BYTES: usize = 64 * 1024;

/// [`replace_literal`] on one thread, for a non-empty `needle`.
fn replace_literal_in<'a>(input: &'a str, needle: &str, replacement: &str) -> Cow<'a, str> {
    // One searcher for the whole input: `str::find` builds a new one per call,
    // which dominated on inputs with a match every few bytes (Gemma's " " → "▁").
    // A match of a UTF-8 needle in UTF-8 text starts and ends on char boundaries.
    let finder = memchr::memmem::Finder::new(needle.as_bytes());
    let mut matches = finder.find_iter(input.as_bytes()).peekable();
    if matches.peek().is_none() {
        return Cow::Borrowed(input);
    }

    let grow = replacement.len().saturating_sub(needle.len());
    let mut output = String::with_capacity(input.len() + input.len() / 4 * grow);
    let mut last = 0;
    for at in matches {
        output.push_str(&input[last..at]);
        output.push_str(replacement);
        last = at + needle.len();
    }
    output.push_str(&input[last..]);
    Cow::Owned(output)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use serde_json::json;

    use super::*;

    #[test]
    fn literal_replace() {
        let repl = Replace::from_config(json!({"String": " "}), "▁".to_string()).unwrap();
        assert_eq!(repl.normalize("a b c"), "a▁b▁c");
    }

    #[test]
    fn literal_no_change_borrowed() {
        let repl = Replace::from_config(json!({"String": "x"}), "y".to_string()).unwrap();
        let out = repl.normalize("abc");
        assert_eq!(out, "abc");
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn regex_replace() {
        let repl = Replace::from_config(json!({"Regex": "\\s+"}), " ".to_string()).unwrap();
        assert_eq!(repl.normalize("hello   world"), "hello world");
    }

    #[test]
    fn accepts_plain_string_pattern() {
        let repl = Replace::from_config(json!("."), " ".to_string()).unwrap();
        assert_eq!(repl.normalize("hello.world"), "hello world");
    }

    #[test]
    fn long_inputs_replace_like_str_replace() {
        // Long enough for the chunked path, with multi-byte chars around the
        // chunk boundaries, and matches at both ends.
        let input = " é 中文 a▁b  😀\t".repeat(30_000) + " ";
        assert!(input.len() > 4 * PARALLEL_MIN);
        for (needle, replacement) in [(" ", "▁"), ("é", "e"), ("中", ""), ("▁b", "B")] {
            let repl = Replace::from_config(json!(needle), replacement.to_string()).unwrap();
            assert_eq!(
                repl.normalize(&input),
                input.replace(needle, replacement),
                "{needle:?}"
            );
        }
        let repl = Replace::from_config(json!("x"), "y".to_string()).unwrap();
        assert!(matches!(repl.normalize(&input), Cow::Borrowed(_)));
    }

    #[test]
    fn literal_empty_pattern_is_not_noop() {
        let repl = Replace::from_config(json!(""), "-".to_string()).unwrap();
        assert_eq!(repl.normalize("ab"), "-a-b-");
    }

    #[test]
    fn regex_replacement_content_is_literal() {
        let repl = Replace::from_config(json!({"Regex": "([a-z]+)"}), "$1".to_string()).unwrap();
        assert_eq!(repl.normalize("abc"), "$1");
    }
}
