//! Shared helpers for the parity/regression test modules: a seeded PRNG,
//! generators for adversarial text, and failure reporting. Everything is
//! deterministic so a failure reproduces from its seed.

/// A seeded PRNG with the draws the generators below use.
pub(crate) struct Rng(fastrand::Rng);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(fastrand::Rng::with_seed(seed))
    }

    /// Uniform in `0..n` (`n > 0`).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        self.0.usize(..n)
    }

    pub(crate) fn chance(&mut self, num: usize, den: usize) -> bool {
        self.below(den) < num
    }

    pub(crate) fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// `(A suffix, B suffix)` pairs for the split cache: after a shared prefix
/// (`"word " * 1000`), B is split right after A on the same thread.
pub(crate) const SPLIT_CACHE_PAIRS: &[(&str, &str)] = &[
    // B's whitespace matches depend on bytes past where A diverges.
    ("   x", "   \n"),
    ("  -", "  \n"),
    ("\n  x", "\n  \n"),
    ("   x", "    \r\n"),
    ("    \u{2}", "     \n"),
    // Controls.
    ("  \n", "  b"),
    ("   ", "  b"),
    ("\n\n\n", "\n\nb"),
    ("  \t", " x"),
];

/// Whitespace the tokenizer regexes treat differently: ASCII blanks, the
/// newline family, and Unicode spaces (`\s` but not ASCII). Repeats weight
/// the draw.
pub(crate) const WHITESPACE: &[&str] = &[
    " ", " ", " ", "  ", "   ", "\t", "\n", "\n", "\r\n", "\r", "\u{0b}", "\u{0c}", "\u{a0}",
    "\u{3000}", "\u{2028}", "\u{2009}", "\u{85}",
];

/// Combining marks, weighted toward ones added in recent Unicode versions
/// (whose data differs between normalization/regex libraries).
pub(crate) const MARKS: &[char] = &[
    '\u{0300}',
    '\u{0301}',
    '\u{0308}',
    '\u{030a}',
    '\u{0341}',
    '\u{0346}',
    '\u{034e}',
    '\u{035b}',
    '\u{0368}',
    '\u{05b4}',
    '\u{0651}',
    '\u{06e0}',
    '\u{07fd}',
    '\u{08cf}',
    '\u{08d3}',
    '\u{093c}',
    '\u{0e49}',
    '\u{1ab0}',
    '\u{1ac1}',
    '\u{1ace}',
    '\u{1aeb}',
    '\u{1dfa}',
    '\u{20d0}',
    '\u{302a}',
    '\u{fe20}',
    '\u{101fd}',
    '\u{10d24}',
    '\u{10eab}',
    '\u{10f46}',
    '\u{11241}',
    '\u{1e4ec}',
    '\u{1e5ee}',
];

/// Short pieces covering every alternative of the tiktoken-family regexes.
const PIECES: &[&str] = &[
    "the",
    "The",
    "THE",
    "word",
    "HTTPRequest",
    "camelCase",
    "iOS",
    "don't",
    "I'LL",
    "'s",
    "'ve",
    "O'Brien",
    "12",
    "3.14",
    "1234567",
    "٣٤",
    "①",
    "Ⅻ",
    "!",
    "...",
    "--",
    "/",
    "//",
    "!\n/",
    "(",
    ")",
    "{\"k\": 1}",
    "<tag>",
    "$x",
    "_",
    "@",
    "#",
    "café",
    "naïve",
    "é\u{301}",
    "中文",
    "你好世界",
    "こんにちは",
    "한국어",
    "Привет",
    "مرحبا",
    "שלום",
    "नमस्ते",
    "😀",
    "👨\u{200d}👩\u{200d}👧",
    "🇺🇸",
    "\u{200b}",
    "\u{feff}",
    "\u{0}",
    "\u{7f}",
    "\u{1}",
    "\u{e000}",
    "\u{10ffff}",
    "ǅ",
    "\u{fffd}",
    "𪛖",
    "𐍈",
    "ꙮ",
    "\u{e0041}",
];

/// A random mix of regex-relevant pieces, whitespace runs and bare marks,
/// roughly `pieces` items long.
pub(crate) fn adversarial_text(rng: &mut Rng, pieces: usize) -> String {
    let mut s = String::new();
    for _ in 0..pieces {
        match rng.below(10) {
            0..=3 => s.push_str(rng.pick(PIECES)),
            4..=6 => s.push_str(rng.pick(WHITESPACE)),
            7 => {
                for _ in 0..1 + rng.below(3) {
                    s.push(rng.pick(MARKS));
                }
            }
            8 => {
                // A long whitespace run of one kind, sometimes ending in a newline.
                let w = rng.pick(WHITESPACE);
                s.push_str(&w.repeat(2 + rng.below(12)));
                if rng.chance(1, 2) {
                    s.push('\n');
                }
            }
            _ => s.push(random_scalar(rng)),
        }
    }
    s
}

/// A random Unicode scalar value, biased toward the BMP.
pub(crate) fn random_scalar(rng: &mut Rng) -> char {
    loop {
        let cp = if rng.chance(3, 4) {
            rng.below(0x1_0000) as u32
        } else {
            rng.below(0x11_0000) as u32
        };
        if let Some(c) = char::from_u32(cp) {
            return c;
        }
    }
}

/// Natural-looking text: words, punctuation and line breaks.
pub(crate) fn prose(rng: &mut Rng, words: usize) -> String {
    const WORDS: &[&str] = &[
        "the",
        "of",
        "and",
        "tokenizer",
        "model",
        "system",
        "prompt",
        "user",
        "said",
        "value",
        "function",
        "return",
        "while",
        "network",
        "data",
        "is",
        "a",
        "to",
        "in",
        "that",
        "网友",
        "数据",
        "résumé",
        "Zürich",
        "2024",
        "3.5",
        "x86_64",
        "e.g.",
        "I'm",
        "they're",
    ];
    let mut s = String::new();
    for i in 0..words {
        if i > 0 {
            s.push_str(match rng.below(20) {
                0 => ", ",
                1 => ". ",
                2 => "\n",
                3 => "\n\n",
                4 => "  ",
                _ => " ",
            });
        }
        s.push_str(rng.pick(WORDS));
    }
    s
}

/// Fail with the first few of `failures`, if there are any.
pub(crate) fn assert_no_failures(what: &str, failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{what}: {} failures, first {}:\n{}",
        failures.len(),
        failures.len().min(10),
        failures[..failures.len().min(10)].join("\n")
    );
}

/// Serializes model loads. hf-hub gives up on a file another thread is
/// still downloading after ~5 s ("Lock acquisition failed"), so concurrent
/// first loads of one model from parallel tests would be flaky.
#[cfg(feature = "hf-hub")]
static LOAD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A fresh fastokens instance of the hub model `name`.
#[cfg(feature = "hf-hub")]
pub(crate) fn load_ours(name: &str) -> crate::Tokenizer {
    let _guard = LOAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    crate::Tokenizer::from_model(name)
        .unwrap_or_else(|e| panic!("{name}: fastokens load failed: {e}"))
}

/// HuggingFace's tokenizer for the hub model `name`.
#[cfg(feature = "hf-hub")]
pub(crate) fn load_hf(name: &str) -> tokenizers::Tokenizer {
    let _guard = LOAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tokenizers::Tokenizer::from_pretrained(name, None)
        .unwrap_or_else(|e| panic!("{name}: HF load failed: {e}"))
}

/// Run `body` in a child copy of the test binary, so a stack overflow (which
/// aborts the whole process) fails only the calling test. `test_path` is the
/// test's full path, e.g. `component_parity_tests::foo`.
pub(crate) fn run_in_child(test_path: &str, body: impl FnOnce()) {
    const CHILD: &str = "FASTOKENS_TEST_CHILD";
    if std::env::var_os(CHILD).is_some() {
        body();
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([test_path, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{test_path} failed in a child process ({}):\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(15)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
