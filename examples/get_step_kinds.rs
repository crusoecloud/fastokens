use std::{collections::BTreeSet, error::Error};

use fastokens::{
    DecoderConfig, DecoderKind, ModelKind, NormalizerConfig, NormalizerKind, PostProcessorConfig,
    PostProcessorKind, PreTokenizerConfig, PreTokenizerKind, TokenizerJson,
};

const MODELS: &[&str] = &[
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "MiniMaxAI/MiniMax-M2.1",
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut model_names = std::env::args().skip(1).collect::<Vec<_>>();
    if model_names.is_empty() {
        model_names = MODELS.iter().map(|s| s.to_string()).collect();
    }

    let api = hf_hub::api::sync::Api::new()?;

    let mut normalizers = BTreeSet::new();
    let mut pre_tokenizers = BTreeSet::new();
    let mut models = BTreeSet::new();
    let mut post_processors = BTreeSet::new();
    let mut decoders = BTreeSet::new();

    for name in model_names {
        eprint!("{name} ... ");
        let repo = api.model(name.clone());
        let json_path = repo.get("tokenizer.json")?;
        let json: TokenizerJson = serde_json::from_str(&std::fs::read_to_string(json_path)?)?;

        if let Some(n) = &json.normalizer {
            collect_normalizers(n, &mut normalizers);
        }
        if let Some(p) = &json.pre_tokenizer {
            collect_pre_tokenizers(p, &mut pre_tokenizers);
        }
        models.insert(ModelKind::from(&json.model));
        if let Some(p) = &json.post_processor {
            collect_post_processors(p, &mut post_processors);
        }
        if let Some(d) = &json.decoder {
            collect_decoders(d, &mut decoders);
        }
        eprintln!("ok");
    }

    print_set("Normalizers", &normalizers);
    print_set("Pre-tokenizers", &pre_tokenizers);
    print_set("Models", &models);
    print_set("Post-processors", &post_processors);
    print_set("Decoders", &decoders);

    Ok(())
}

fn print_set<T: std::fmt::Display>(label: &str, set: &BTreeSet<T>) {
    println!("{label}:");
    if set.is_empty() {
        println!("  (none)");
    } else {
        for kind in set {
            println!("  {kind}");
        }
    }
    println!();
}

fn collect_normalizers(n: &NormalizerConfig, set: &mut BTreeSet<NormalizerKind>) {
    set.insert(NormalizerKind::from(n));
    if let NormalizerConfig::Sequence { normalizers } = n {
        for child in normalizers {
            collect_normalizers(child, set);
        }
    }
}

fn collect_pre_tokenizers(p: &PreTokenizerConfig, set: &mut BTreeSet<PreTokenizerKind>) {
    set.insert(PreTokenizerKind::from(p));
    if let PreTokenizerConfig::Sequence { pretokenizers } = p {
        for child in pretokenizers {
            collect_pre_tokenizers(child, set);
        }
    }
}

fn collect_post_processors(p: &PostProcessorConfig, set: &mut BTreeSet<PostProcessorKind>) {
    set.insert(PostProcessorKind::from(p));
    if let PostProcessorConfig::Sequence { processors } = p {
        for child in processors {
            collect_post_processors(child, set);
        }
    }
}

fn collect_decoders(d: &DecoderConfig, set: &mut BTreeSet<DecoderKind>) {
    set.insert(DecoderKind::from(d));
    if let DecoderConfig::Sequence { decoders } = d {
        for child in decoders {
            collect_decoders(child, set);
        }
    }
}
