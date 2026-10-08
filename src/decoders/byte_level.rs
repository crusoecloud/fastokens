use crate::pre_tokenizers::byte_level::BYTE_TO_CHAR;

/// Reverse mapping: Unicode char → original byte value, if the char is in the
/// GPT-2 table.
///
/// All chars in `BYTE_TO_CHAR` are in the range U+0000..U+0143 (max codepoint
/// 323), so a flat 324-element array gives O(1) lookup. Not every char in that
/// range is in the table (e.g. space and `\n` are not).
const CHAR_TO_BYTE: [Option<u8>; 324] = build_char_to_byte();

const fn build_char_to_byte() -> [Option<u8>; 324] {
    let mut table = [None; 324];
    let mut i = 0u16;
    while i < 256 {
        let ch = BYTE_TO_CHAR[i as usize];
        table[ch as usize] = Some(i as u8);
        i += 1;
    }
    table
}

/// The byte `c` stands for in the GPT-2 table, if it is in the table.
fn char_to_byte(c: char) -> Option<u8> {
    CHAR_TO_BYTE.get(c as usize).copied().flatten()
}

/// ByteLevel decoder: reverses the GPT-2 byte-to-unicode mapping.
#[derive(Debug)]
pub struct ByteLevelDecoder;

impl ByteLevelDecoder {
    /// Apply byte-level decoding to a list of token strings.
    ///
    /// Like HF `tokenizers`, each token is mapped back to bytes only if every
    /// char in it is in the reverse GPT-2 table; otherwise (e.g. an added
    /// token such as `<｜begin▁of▁sentence｜>`, or one containing a plain
    /// space) the token contributes its own UTF-8 unchanged. The bytes of all
    /// tokens are then interpreted as UTF-8.
    pub fn decode_chain(&self, tokens: Vec<String>) -> Vec<String> {
        let mut bytes: Vec<u8> = Vec::with_capacity(tokens.iter().map(String::len).sum());
        for token in &tokens {
            let start = bytes.len();
            for c in token.chars() {
                match char_to_byte(c) {
                    Some(b) => bytes.push(b),
                    None => {
                        bytes.truncate(start);
                        bytes.extend_from_slice(token.as_bytes());
                        break;
                    }
                }
            }
        }
        vec![String::from_utf8_lossy(&bytes).into_owned()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ascii() {
        let dec = ByteLevelDecoder;
        let result = dec.decode_chain(vec!["Hello".to_string()]);
        assert_eq!(result, vec!["Hello"]);
    }

    #[test]
    fn roundtrip_space() {
        let dec = ByteLevelDecoder;
        // GPT-2 maps space (0x20) to Ġ (U+0120)
        let result = dec.decode_chain(vec!["\u{120}Hello".to_string()]);
        assert_eq!(result, vec![" Hello"]);
    }

    #[test]
    fn roundtrip_multibyte() {
        let dec = ByteLevelDecoder;
        // Euro sign €: UTF-8 bytes [0xE2, 0x82, 0xAC]
        // BYTE_TO_CHAR maps these to specific unicode chars
        let encoded: String = [0xE2u8, 0x82, 0xAC]
            .iter()
            .map(|&b| BYTE_TO_CHAR[b as usize])
            .collect();
        let result = dec.decode_chain(vec![encoded]);
        assert_eq!(result, vec!["€"]);
    }

    #[test]
    fn non_gpt2_chars_preserved() {
        let dec = ByteLevelDecoder;
        // DeepSeek uses ｜ (U+FF5C) and ▁ (U+2581) in token strings
        // like <｜begin▁of▁sentence｜>. These are outside the GPT-2
        // byte-to-char table and must pass through unchanged.
        let result = dec.decode_chain(vec![
            "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>".to_string(),
        ]);
        assert_eq!(result, vec!["<｜begin▁of▁sentence｜>"]);
    }

    #[test]
    fn token_with_any_char_outside_table_is_kept_whole() {
        // Space and '\n' are below U+0144 but not in the table: a token holding
        // one keeps its own UTF-8, including its table chars (Ġ, Ċ), as in HF.
        let dec = ByteLevelDecoder;
        let result = dec.decode_chain(vec![
            "\u{120}hello world".to_string(),
            "\u{10a}\u{120}中".to_string(),
            " ".to_string(),
            "\u{120}x".to_string(),
        ]);
        assert_eq!(result, vec!["\u{120}hello world\u{10a}\u{120}中  x"]);
    }
}
