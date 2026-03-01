use crate::pre_tokenizers::byte_level::BYTE_TO_CHAR;

/// Reverse mapping: Unicode char → original byte value.
///
/// All chars in `BYTE_TO_CHAR` are in the range U+0000..U+0143 (max codepoint
/// 323), so a flat 324-element array gives O(1) lookup.
const CHAR_TO_BYTE: [u8; 324] = build_char_to_byte();

const fn build_char_to_byte() -> [u8; 324] {
    let mut table = [0u8; 324];
    let mut i = 0u16;
    while i < 256 {
        let ch = BYTE_TO_CHAR[i as usize];
        table[ch as usize] = i as u8;
        i += 1;
    }
    table
}

/// ByteLevel decoder: reverses the GPT-2 byte-to-unicode mapping.
#[derive(Debug)]
pub struct ByteLevelDecoder;

impl ByteLevelDecoder {
    /// Apply byte-level decoding to a list of token strings.
    ///
    /// Joins all tokens, maps each char back to its byte value using the
    /// reverse GPT-2 table, and interprets the bytes as UTF-8.
    pub fn decode_chain(&self, tokens: Vec<String>) -> Vec<String> {
        let joined: String = tokens.into_iter().collect();
        let bytes: Vec<u8> = joined
            .chars()
            .map(|c| {
                let cp = c as usize;
                if cp < CHAR_TO_BYTE.len() {
                    CHAR_TO_BYTE[cp]
                } else {
                    // Characters outside the GPT-2 table are kept as-is
                    // (shouldn't happen for well-formed tokens).
                    b'?'
                }
            })
            .collect();
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
}
