use candle_core::{Device, Result as CandleResult, Tensor};
use candle_transformers::models::quantized_qwen2 as candle_qwen2;
use std::sync::Mutex;
use tokenizers::Tokenizer;

use crate::error::{PipelineError, Result};
use crate::loaders::{GgufModelLoader, TokenizerLoader};

/// ReaderLM-v2 model for HTML to Markdown/JSON conversion.
///
/// Based on Qwen2.5-1.5B, fine-tuned by Jina AI for HTML parsing tasks.
pub struct ReaderLM {
    weights: Mutex<candle_qwen2::ModelWeights>,
    max_seq_len: usize,
    device: Device,
}

impl ReaderLM {
    /// HuggingFace repo for tokenizer and config.
    const TOKENIZER_REPO: &'static str = "jinaai/ReaderLM-v2";

    /// GGUF weights repo (Q4_K_M quantization).
    const GGUF_REPO: &'static str = "mradermacher/ReaderLM-v2-GGUF";
    const GGUF_FILE: &'static str = "ReaderLM-v2.Q4_K_M.gguf";

    pub(crate) async fn from_hf(device: &Device) -> Result<Self> {
        let model_loader = GgufModelLoader::new(Self::GGUF_REPO, Self::GGUF_FILE);
        let (mut file, content) = model_loader.load().await?;

        let max_seq_len = content
            .metadata
            .get("qwen2.context_length")
            .ok_or_else(|| {
                PipelineError::Unexpected(
                    "Missing 'qwen2.context_length' in ReaderLM model metadata".to_string(),
                )
            })?
            .to_u32()? as usize;

        let weights = candle_qwen2::ModelWeights::from_gguf(content, &mut file, device)?;

        Ok(Self {
            weights: Mutex::new(weights),
            max_seq_len,
            device: device.clone(),
        })
    }

    pub(crate) async fn get_tokenizer() -> Result<Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new(Self::TOKENIZER_REPO, "tokenizer.json");
        tokenizer_loader.load().await
    }

    pub(crate) fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// Forward pass returning logits for the last token position.
    pub(crate) fn forward(&self, input_ids: &Tensor, position: usize) -> CandleResult<Tensor> {
        let mut weights = self.weights.lock().unwrap();
        weights.forward(input_ids, position)
    }
}
