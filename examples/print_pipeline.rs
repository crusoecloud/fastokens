use std::error::Error;

use fastokens::Tokenizer;

fn main() -> Result<(), Box<dyn Error>> {
    let Some(model) = std::env::args().nth(1) else {
        return Err("specify model argument".into());
    };

    let tok = Tokenizer::from_model(&model)?;

    println!("Model: {model}\n");

    println!("Normalizer:      {}", opt_debug(tok.normalizer()));
    println!("Pre-tokenizer:   {}", opt_debug(tok.pre_tokenizer()));
    println!("Model:           {:?}", tok.model());

    Ok(())
}

fn opt_debug<T: std::fmt::Debug>(v: Option<&T>) -> String {
    match v {
        Some(v) => format!("{v:#?}"),
        None => "(none)".to_string(),
    }
}
