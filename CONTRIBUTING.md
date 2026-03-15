# Contributing to fastokens

Thank you for your interest in contributing. The notes below cover the development workflow and testing requirements.

## Building

```sh
cargo build
```

To build without the optional `pcre2` C dependency (pure-Rust build):

```sh
cargo build --no-default-features
```

## Testing
Run the test suite to catch regressions:

```sh
# Default build — pcre2 JIT fast path enabled
cargo test

# Pure-Rust build — fancy-regex fallback, no libpcre2-8 required
cargo test --no-default-features
```

## Code style

After making any changes to Rust source files, run the formatter before committing:

```sh
cargo fmt
```

## Feature flags

| Feature | Default | Notes |
|---------|---------|-------|
| `pcre2` | on | Links to `libpcre2-8` via `pcre2-sys`. Disable with `--no-default-features` if the system library is unavailable. |
