use fancy_regex::Regex;
use serde::Deserialize;

use crate::pre_tokenized::{PreTokenizedString, Split as PtSplit};

use super::Error;

/// GPT-2 byte-to-unicode mapping table.
///
/// Bytes that are "nice" printable ASCII / Latin-1 characters map to the
/// codepoint with the same value. The remaining 68 bytes map to codepoints
/// starting at U+0100 (Ā).
const BYTE_TO_CHAR: [char; 256] = build_byte_to_char();

const fn build_byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next: u32 = 256;
    let mut i: u16 = 0;
    while i < 256 {
        let b = i as u8;
        let nice = (b >= b'!' && b <= b'~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
        let cp = if nice {
            i as u32
        } else {
            let cp = next;
            next += 1;
            cp
        };
        // All codepoints 0..=323 are valid Unicode scalars.
        table[i as usize] = match char::from_u32(cp) {
            Some(c) => c,
            None => panic!("invalid codepoint"),
        };
        i += 1;
    }
    table
}

/// Append the byte-level encoding of `s` to `out`.
fn encode_bytes_into(s: &str, out: &mut String) {
    for &b in s.as_bytes() {
        out.push(BYTE_TO_CHAR[b as usize]);
    }
}

/// Encode a string by mapping each byte of its UTF-8 representation to the
/// corresponding visible character from the GPT-2 byte-to-unicode table.
#[cfg(test)]
fn encode_bytes(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    encode_bytes_into(s, &mut out);
    out
}

/// GPT-2 pretokenization regex.
///
/// Matches contractions, letter runs, number runs, punctuation runs, and
/// whitespace — in that priority order.
const GPT2_PATTERN: &str = concat!(
    r"'(?i:[sdmt])",
    r"|'(?i:ll|ve|re)",
    r"| ?\p{L}+",
    r"| ?\p{N}+",
    r"| ?[^\s\p{L}\p{N}]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

fn default_true() -> bool {
    true
}

/// Raw deserialization helper for [`ByteLevel`].
#[derive(Deserialize)]
struct ByteLevelRaw {
    #[serde(default = "default_true")]
    add_prefix_space: bool,
    #[serde(default = "default_true")]
    trim_offsets: bool,
    #[serde(default = "default_true")]
    use_regex: bool,
}

/// A compiled ByteLevel pre-tokenizer.
///
/// Optionally splits input using the GPT-2 regex, then maps
/// every byte of the resulting UTF-8 text to a visible
/// Unicode character.
///
/// Constructed once from config fields (typically via
/// [`PreTokenizerConfig::ByteLevel`]), then reused across
/// many inputs.
///
/// [`PreTokenizerConfig::ByteLevel`]:
///     crate::PreTokenizerConfig::ByteLevel
#[derive(Clone, Debug, Deserialize)]
#[serde(try_from = "ByteLevelRaw")]
pub struct ByteLevel {
    regex: Option<Regex>,
    add_prefix_space: bool,
    /// Stored for config fidelity; not used in splitting.
    #[allow(dead_code)]
    trim_offsets: bool,
}

impl TryFrom<ByteLevelRaw> for ByteLevel {
    type Error = fancy_regex::Error;

    fn try_from(raw: ByteLevelRaw) -> Result<Self, fancy_regex::Error> {
        let regex = if raw.use_regex {
            Some(Regex::new(GPT2_PATTERN)?)
        } else {
            None
        };
        Ok(Self {
            regex,
            add_prefix_space: raw.add_prefix_space,
            trim_offsets: raw.trim_offsets,
        })
    }
}

impl ByteLevel {
    /// Build a [`ByteLevel`] from config fields.
    #[cfg(test)]
    fn from_config(
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    ) -> Result<Self, serde_json::Error> {
        serde_json::from_value(serde_json::json!({
            "add_prefix_space": add_prefix_space,
            "trim_offsets": trim_offsets,
            "use_regex": use_regex,
        }))
    }

    /// Pre-tokenize in place using byte-level encoding.
    ///
    /// For each text split in `pts`: optionally prepend a space, split
    /// by the GPT-2 regex, and byte-encode each piece. Splits with a
    /// pre-assigned token ID are byte-encoded but otherwise passed
    /// through.
    pub fn pre_tokenize(&self, pts: &mut PreTokenizedString) -> Result<(), Error> {
        let old_buf = pts.buffer();
        let mut new_buf = String::with_capacity(old_buf.len().saturating_mul(2));
        let mut new_splits = Vec::with_capacity(pts.splits().len().saturating_mul(4));

        for split in pts.splits() {
            let text = pts.split_text(split);

            if split.token_id.is_some() {
                // Added token: byte-encode but keep the token ID.
                let start = new_buf.len();
                encode_bytes_into(text, &mut new_buf);
                let end = new_buf.len();
                new_splits.push(PtSplit {
                    range: start..end,
                    token_id: split.token_id,
                });
                continue;
            }

            if text.is_empty() {
                continue;
            }

            let prefixed;
            let text = if self.add_prefix_space && !text.starts_with(' ') {
                prefixed = format!(" {text}");
                prefixed.as_str()
            } else {
                text
            };

            match &self.regex {
                Some(re) => {
                    for m in re.find_iter(text) {
                        let m = m?;
                        if !m.as_str().is_empty() {
                            let start = new_buf.len();
                            encode_bytes_into(m.as_str(), &mut new_buf);
                            let end = new_buf.len();
                            new_splits.push(PtSplit {
                                range: start..end,
                                token_id: None,
                            });
                        }
                    }
                }
                None => {
                    let start = new_buf.len();
                    encode_bytes_into(text, &mut new_buf);
                    let end = new_buf.len();
                    if start < end {
                        new_splits.push(PtSplit {
                            range: start..end,
                            token_id: None,
                        });
                    }
                }
            }
        }

        pts.set_buffer(new_buf, new_splits);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    // ── Byte-to-char table ─────────────────────────────

    #[test]
    fn table_has_256_unique_chars() {
        let mut seen = HashSet::new();
        for &c in &BYTE_TO_CHAR {
            assert!(seen.insert(c), "duplicate char: {c:?}");
        }
    }

    #[test]
    fn nice_bytes_map_to_themselves() {
        assert_eq!(BYTE_TO_CHAR[b'!' as usize], '!');
        assert_eq!(BYTE_TO_CHAR[b'A' as usize], 'A');
        assert_eq!(BYTE_TO_CHAR[b'z' as usize], 'z');
        assert_eq!(BYTE_TO_CHAR[b'~' as usize], '~');
        assert_eq!(BYTE_TO_CHAR[0xA1], '\u{A1}');
        assert_eq!(BYTE_TO_CHAR[0xAC], '\u{AC}');
        assert_eq!(BYTE_TO_CHAR[0xAE], '\u{AE}');
        assert_eq!(BYTE_TO_CHAR[0xFF], '\u{FF}');
    }

    #[test]
    fn remapped_bytes_start_at_256() {
        // Byte 0 is the first non-nice byte.
        assert_eq!(BYTE_TO_CHAR[0], '\u{100}');
        // Space (32) → U+0120 (Ġ).
        assert_eq!(BYTE_TO_CHAR[b' ' as usize], 'Ġ');
        // Newline (10) → U+010A (Ċ).
        assert_eq!(BYTE_TO_CHAR[b'\n' as usize], 'Ċ');
    }

    #[test]
    fn non_nice_count_is_68() {
        let count = BYTE_TO_CHAR.iter().filter(|&&c| c as u32 >= 256).count();
        assert_eq!(count, 68);
    }

    // ── encode_bytes ───────────────────────────────────

    #[test]
    fn encode_ascii() {
        assert_eq!(encode_bytes("Hello"), "Hello");
    }

    #[test]
    fn encode_space() {
        assert_eq!(encode_bytes(" "), "\u{120}");
    }

    #[test]
    fn encode_multibyte_utf8() {
        // Euro sign €: UTF-8 bytes [0xE2, 0x82, 0xAC].
        let encoded = encode_bytes("\u{20AC}");
        assert_eq!(encoded.chars().count(), 3);
        assert_eq!(
            encoded,
            format!(
                "{}{}{}",
                BYTE_TO_CHAR[0xE2], BYTE_TO_CHAR[0x82], BYTE_TO_CHAR[0xAC],
            )
        );
    }

    // ── pre_tokenize ─────────────────────────────────

    /// Helper: run byte-level pre-tokenize on a plain string and return the
    /// resulting split texts.
    fn run(bl: &ByteLevel, input: &str) -> Vec<String> {
        let mut pts = PreTokenizedString::from_text(input);
        bl.pre_tokenize(&mut pts).unwrap();
        pts.splits()
            .iter()
            .map(|s| pts.split_text(s).to_string())
            .collect()
    }

    #[test]
    fn simple_words() {
        let bl = ByteLevel::from_config(false, true, true).unwrap();
        let result = run(&bl, "Hello world");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "Hello");
        assert_eq!(result[1], format!("{}world", BYTE_TO_CHAR[b' ' as usize]));
    }

    #[test]
    fn contractions() {
        let bl = ByteLevel::from_config(false, true, true).unwrap();
        let result = run(&bl, "I'm");
        assert_eq!(result, vec!["I", "'m"]);
    }

    #[test]
    fn numbers_and_punctuation() {
        let bl = ByteLevel::from_config(false, true, true).unwrap();
        let result = run(&bl, "price: $100");
        assert!(result.len() >= 3);
    }

    // ── add_prefix_space ──────────────────────────────

    #[test]
    fn prefix_space_added() {
        let bl = ByteLevel::from_config(true, true, true).unwrap();
        let result = run(&bl, "Hello");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], format!("{}Hello", BYTE_TO_CHAR[b' ' as usize]));
    }

    #[test]
    fn prefix_space_not_doubled() {
        let bl = ByteLevel::from_config(true, true, true).unwrap();
        let result = run(&bl, " Hello");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], format!("{}Hello", BYTE_TO_CHAR[b' ' as usize]));
    }

    // ── use_regex = false ─────────────────────────────

    #[test]
    fn no_regex_single_segment() {
        let bl = ByteLevel::from_config(false, true, false).unwrap();
        let result = run(&bl, "Hello world");
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0],
            format!("Hello{}world", BYTE_TO_CHAR[b' ' as usize]),
        );
    }

    // ── Edge cases ────────────────────────────────────

    #[test]
    fn empty_input() {
        let bl = ByteLevel::from_config(false, true, true).unwrap();
        let result = run(&bl, "");
        assert!(result.is_empty());
    }

    #[test]
    fn empty_input_with_prefix_space() {
        let bl = ByteLevel::from_config(true, true, true).unwrap();
        let result = run(&bl, "");
        assert!(result.is_empty());
    }

    #[test]
    fn all_whitespace() {
        let bl = ByteLevel::from_config(false, true, true).unwrap();
        let result = run(&bl, "   ");
        assert!(!result.is_empty());
    }

    #[test]
    fn non_ascii_input() {
        let bl = ByteLevel::from_config(false, true, true).unwrap();
        // 猫 is 3 UTF-8 bytes → 3 chars in output.
        let result = run(&bl, "猫");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].chars().count(), 3);
    }

    // ── Added-token passthrough ───────────────────────

    #[test]
    fn added_token_splits_preserved() {
        let bl = ByteLevel::from_config(false, true, true).unwrap();
        let buffer = "hello<sep>world".to_string();
        let splits = vec![
            PtSplit {
                range: 0..5,
                token_id: None,
            },
            PtSplit {
                range: 5..10,
                token_id: Some(42),
            },
            PtSplit {
                range: 10..15,
                token_id: None,
            },
        ];
        let mut pts = PreTokenizedString::new(buffer, splits);
        bl.pre_tokenize(&mut pts).unwrap();

        // The added-token split should still have its token_id.
        let added = pts
            .splits()
            .iter()
            .find(|s| s.token_id == Some(42))
            .expect("added token split missing");
        // Its text should be byte-encoded.
        assert_eq!(pts.split_text(added), encode_bytes("<sep>"));
    }

    // ── Serde ─────────────────────────────────────────

    #[test]
    fn deserialize_default_config() {
        let bl: ByteLevel = serde_json::from_str("{}").unwrap();
        assert!(bl.regex.is_some());
        assert!(bl.add_prefix_space);
        assert!(bl.trim_offsets);
    }

    #[test]
    fn deserialize_no_regex() {
        let bl: ByteLevel = serde_json::from_str(r#"{"use_regex":false}"#).unwrap();
        assert!(bl.regex.is_none());
    }
}
