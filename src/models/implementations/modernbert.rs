//! High-performance ModernBERT implementation.
//!
//! ModernBERT is a modernized bidirectional encoder-only Transformer model with:
//! - Sliding window attention for efficient long context processing
//! - Global attention layers for capturing long-range dependencies
//! - Multiple task heads (masked LM, sequence classification)
//! - Memory efficient inference
//!
//! # Quick Start
//! ```rust
//! use candle_core::Device;
//! use candle_nn::VarBuilder;
//! use std::fs::File;
//!
//! // For masked language modeling
//! // let model_mlm = ModernBertForMaskedLM::load(vb.clone(), &config)?;
//! // let output_mlm = model_mlm.forward(&input_tokens, &attention_mask)?;
//!
//! // For sequence classification
//! // let model_clf = ModernBertForSequenceClassification::load(vb.clone(), &config)?;
//! // let output_clf = model_clf.forward(&input_tokens, &attention_mask)?;
//!
//! // For zero-shot classification
//! // let model = ZeroShotModernBertModel::new(ZeroShotModernBertSize::Large)?;
//! // let results = model.predict(&tokenizer, premise, &candidate_labels)?;
//!
//! // For sentiment analysis
//! // let model = SentimentModernBertModel::new(SentimentModernBertSize::Base)?;
//! // let sentiment = model.predict(&tokenizer, text)?;
//! ```

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm_no_bias, linear, linear_no_bias, ops::softmax, Embedding, LayerNorm,
    Linear, Module, VarBuilder,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

// Constants
const NEG_INF: f32 = f32::NEG_INFINITY;
const MIN_VALUE_F64: f64 = f32::MIN as f64;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
    #[serde(flatten)]
    pub classifier_config: Option<ClassifierConfig>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Copy, Default)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierPooling {
    #[default]
    CLS,
    MEAN,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ClassifierConfig {
    pub id2label: HashMap<String, String>,
    pub label2id: HashMap<String, String>,
    pub classifier_pooling: ClassifierPooling,
}

/// Rotary Position Embedding (RoPE) implementation.
#[derive(Debug, Clone)]
struct RoPE {
    sin: Tensor,
    cos: Tensor,
}

impl RoPE {
    fn new(dtype: DType, config: &Config, rope_theta: f64, device: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| (1.0 / rope_theta.powf(i as f64 / dim as f64)) as f32)
            .collect();

        // Capture length before the vector is moved into `from_vec`.
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(dtype)?;
        let max_seq_len = config.max_position_embeddings;
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let angles = positions.matmul(&inv_freq)?;

        Ok(Self {
            sin: angles.sin()?,
            cos: angles.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?;
        Ok((q_embed, k_embed))
    }
}

/// Multi-head attention with sliding window support.
#[derive(Debug, Clone)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
    rope: Arc<RoPE>,
}

impl Attention {
    fn load(vb: VarBuilder, config: &Config, rope: Arc<RoPE>) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let qkv = linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("Wqkv"))?;
        let proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("Wo"))?;

        Ok(Self {
            qkv,
            proj,
            num_attention_heads,
            attention_head_size,
            rope,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = hidden_states.dims3()?;

        let qkv = hidden_states
            .apply(&self.qkv)?
            .reshape((
                batch,
                seq_len,
                3,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        let (q, k) = self.rope.apply(&q, &k)?;

        let scale = (self.attention_head_size as f64).powf(-0.5);
        let q = (q * scale)?;

        let attention_scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attention_scores = attention_scores.broadcast_add(attention_mask)?;
        let attention_probs = softmax(&attention_scores, D::Minus1)?;

        let context = attention_probs.matmul(&v)?;
        let output = context
            .transpose(1, 2)?
            .reshape((batch, seq_len, hidden_size))?
            .apply(&self.proj)?;

        Ok(output)
    }
}

/// Feed-forward network with GeGLU activation.
#[derive(Debug, Clone)]
struct FeedForward {
    wi: Linear,
    wo: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let wi = linear_no_bias(
            config.hidden_size,
            config.intermediate_size * 2,
            vb.pp("Wi"),
        )?;
        let wo = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("Wo"))?;
        Ok(Self { wi, wo })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.wi)?;
        let chunks = xs.chunk(2, D::Minus1)?;
        let output = (&chunks[0].gelu_erf()? * &chunks[1])?.apply(&self.wo)?;
        Ok(output)
    }
}

/// Single transformer layer.
#[derive(Debug, Clone)]
struct TransformerLayer {
    attention: Attention,
    feed_forward: FeedForward,
    attention_norm: Option<LayerNorm>,
    ffn_norm: LayerNorm,
    uses_local_attention: bool,
}

impl TransformerLayer {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rope: Arc<RoPE>,
        uses_local_attention: bool,
    ) -> Result<Self> {
        let attention = Attention::load(vb.pp("attn"), config, rope)?;
        let feed_forward = FeedForward::load(vb.pp("mlp"), config)?;

        let attention_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("attn_norm"),
        )
        .ok();

        let ffn_norm =
            layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("mlp_norm"))?;

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            uses_local_attention,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        global_attention_mask: &Tensor,
        local_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let mut normed = hidden_states.clone();

        if let Some(norm) = &self.attention_norm {
            normed = normed.apply(norm)?;
        }

        let attention_mask = if self.uses_local_attention {
            &global_attention_mask.broadcast_add(local_attention_mask)?
        } else {
            global_attention_mask
        };

        let attention_output = self.attention.forward(&normed, attention_mask)?;
        let hidden_states = (residual + attention_output)?;

        let ffn_output = hidden_states
            .apply(&self.ffn_norm)?
            .apply(&self.feed_forward)?;
        hidden_states + ffn_output
    }
}

/// Task-specific head for masked language modeling.
#[derive(Debug, Clone)]
struct MaskedLMHead {
    dense: Linear,
    norm: LayerNorm,
    decoder: Linear,
}

impl MaskedLMHead {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("head.dense"))?;
        let norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("head.norm"),
        )?;

        let decoder_weights = vb.get(
            (config.vocab_size, config.hidden_size),
            "model.embeddings.tok_embeddings.weight",
        )?;
        let decoder_bias = vb.get(config.vocab_size, "decoder.bias")?;
        let decoder = Linear::new(decoder_weights, Some(decoder_bias));

        Ok(Self {
            dense,
            norm,
            decoder,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        hidden_states
            .apply(&self.dense)?
            .gelu_erf()?
            .apply(&self.norm)?
            .apply(&self.decoder)
    }
}

/// Task-specific head for sequence classification.
#[derive(Debug, Clone)]
struct ClassificationHead {
    dense: Linear,
    norm: LayerNorm,
    classifier: Linear,
    pooling: ClassifierPooling,
}

impl ClassificationHead {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("head.dense"))?;
        let norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("head.norm"),
        )?;

        let num_labels = config
            .classifier_config
            .as_ref()
            .map_or(0, |c| c.id2label.len());
        let classifier = linear(config.hidden_size, num_labels, vb.pp("classifier"))?;

        let pooling = config
            .classifier_config
            .as_ref()
            .map_or(ClassifierPooling::CLS, |c| c.classifier_pooling);

        Ok(Self {
            dense,
            norm,
            classifier,
            pooling,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let pooled = match self.pooling {
            ClassifierPooling::CLS => hidden_states.i((.., 0, ..))?,
            ClassifierPooling::MEAN => {
                let mask = attention_mask.unsqueeze(D::Minus1)?.to_dtype(DType::F32)?;
                let sum_hidden = hidden_states.broadcast_mul(&mask)?.sum(1)?;
                let sum_mask = attention_mask.sum_keepdim(1)?.to_dtype(DType::F32)?;
                sum_hidden.broadcast_div(&sum_mask)?
            }
        };

        pooled
            .apply(&self.dense)?
            .gelu_erf()?
            .apply(&self.norm)?
            .apply(&self.classifier)
    }
}

/// Shared model weights that can be used across multiple model instances.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    embeddings: Embedding,
    embedding_norm: LayerNorm,
    layers: Vec<TransformerLayer>,
    final_norm: LayerNorm,
    local_attention_size: usize,
    device: Device,
    dtype: DType,
}

impl ModelWeights {
    /// Load model weights from safetensors or other format.
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embeddings.tok_embeddings"),
        )?;

        let embedding_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("model.embeddings.norm"),
        )?;

        let global_rope = Arc::new(RoPE::new(
            vb.dtype(),
            config,
            config.global_rope_theta,
            vb.device(),
        )?);

        let local_rope = Arc::new(RoPE::new(
            vb.dtype(),
            config,
            config.local_rope_theta,
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let uses_local_attention = layer_idx % config.global_attn_every_n_layers != 0;
            let rope = if uses_local_attention {
                local_rope.clone()
            } else {
                global_rope.clone()
            };

            layers.push(TransformerLayer::load(
                vb.pp(format!("model.layers.{layer_idx}")),
                config,
                rope,
                uses_local_attention,
            )?);
        }

        let final_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("model.final_norm"),
        )?;

        Ok(Self {
            embeddings,
            embedding_norm,
            layers,
            final_norm,
            local_attention_size: config.local_attention,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Create global attention mask from padding mask.
    fn create_global_attention_mask(&self, mask: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = mask.dims2()?;

        let expanded_mask = mask
            .unsqueeze(1)?
            .unsqueeze(2)?
            .expand((batch_size, 1, seq_len, seq_len))?
            .to_dtype(self.dtype)?;

        let inverted_mask = (1.0 - expanded_mask)?;
        (inverted_mask * MIN_VALUE_F64)?.to_dtype(self.dtype)
    }

    /// Create local sliding window attention mask.
    fn create_local_attention_mask(&self, seq_len: usize) -> Result<Tensor> {
        let max_distance = self.local_attention_size / 2;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if (j as i32 - i as i32).abs() > max_distance as i32 {
                        NEG_INF
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)
    }

    /// Forward pass through the base model.
    fn forward_base(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;

        let global_attention_mask = self.create_global_attention_mask(attention_mask)?;
        let local_attention_mask = self.create_local_attention_mask(seq_len)?;

        let mut hidden_states = input_ids
            .apply(&self.embeddings)?
            .apply(&self.embedding_norm)?;

        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                &global_attention_mask,
                &local_attention_mask,
            )?;
        }

        hidden_states.apply(&self.final_norm)
    }
}

/// High-level ModernBERT base model interface.
#[derive(Debug, Clone)]
pub struct ModernBertModel {
    weights: Arc<ModelWeights>,
}

impl ModernBertModel {
    /// Load a ModernBERT base model.
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let weights = Arc::new(ModelWeights::load(vb, config)?);
        Ok(Self { weights })
    }

    /// Forward pass returning hidden states.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs with shape `(batch_size, sequence_length)`
    /// * `attention_mask` - Attention mask with shape `(batch_size, sequence_length)` (1 for unmasked, 0 for masked/padded)
    ///
    /// # Returns
    /// Hidden states with shape `(batch_size, sequence_length, hidden_size)`
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.weights.forward_base(input_ids, attention_mask)
    }

    /// Get model information.
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            num_layers: self.weights.layers.len(),
            device: self.weights.device.clone(),
            dtype: self.weights.dtype,
        }
    }
}

/// ModernBERT model for masked language modeling.
#[derive(Debug, Clone)]
pub struct ModernBertForMaskedLM {
    weights: Arc<ModelWeights>,
    head: MaskedLMHead,
}

impl ModernBertForMaskedLM {
    /// Load a ModernBERT model for masked LM.
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let weights = Arc::new(ModelWeights::load(vb.clone(), config)?);
        let head = MaskedLMHead::load(vb, config)?;
        Ok(Self { weights, head })
    }

    /// Forward pass for masked language modeling.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs with shape `(batch_size, sequence_length)`
    /// * `attention_mask` - Attention mask with shape `(batch_size, sequence_length)` (1 for unmasked, 0 for masked/padded)
    ///
    /// # Returns
    /// Logits over vocabulary with shape `(batch_size, sequence_length, vocab_size)`
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden_states = self.weights.forward_base(input_ids, attention_mask)?;
        self.head.forward(&hidden_states)
    }

    /// Get model information.
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            num_layers: self.weights.layers.len(),
            device: self.weights.device.clone(),
            dtype: self.weights.dtype,
        }
    }
}

/// ModernBERT model for sequence classification.
#[derive(Debug, Clone)]
pub struct ModernBertForSequenceClassification {
    weights: Arc<ModelWeights>,
    head: ClassificationHead,
}

impl ModernBertForSequenceClassification {
    /// Load a ModernBERT model for sequence classification.
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let weights = Arc::new(ModelWeights::load(vb.clone(), config)?);
        let head = ClassificationHead::load(vb, config)?;
        Ok(Self { weights, head })
    }

    /// Forward pass for sequence classification.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs with shape `(batch_size, sequence_length)`
    /// * `attention_mask` - Attention mask with shape `(batch_size, sequence_length)` (1 for unmasked, 0 for masked/padded)
    ///
    /// # Returns
    /// Classification logits with shape `(batch_size, num_labels)`
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden_states = self.weights.forward_base(input_ids, attention_mask)?;
        self.head.forward(&hidden_states, attention_mask)
    }

    /// Get model information.
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            num_layers: self.weights.layers.len(),
            device: self.weights.device.clone(),
            dtype: self.weights.dtype,
        }
    }
}

/// Model information structure.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub device: Device,
    pub dtype: DType,
}

/*
Pipeline Implementations
*/

use anyhow::{Error as E, Result as AnyhowResult};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

/// Available ModernBERT model sizes used for tasks like fill-mask.
#[derive(Debug, Clone, Copy)]
pub enum ModernBertSize {
    Base,
    Large,
}

impl std::fmt::Display for ModernBertSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ModernBertSize::Base => "modernbert-base",
            ModernBertSize::Large => "modernbert-large",
        };
        write!(f, "{name}")
    }
}

impl crate::core::ModelOptions for ModernBertSize {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

/// Fill-mask model using ModernBERT.
#[derive(Clone)]
pub struct FillMaskModernBertModel {
    model: ModernBertForMaskedLM,
    device: Device,
}

impl FillMaskModernBertModel {
    pub fn new(size: ModernBertSize, device: Device) -> AnyhowResult<Self> {
        let model_id = match size {
            ModernBertSize::Base => "answerdotai/ModernBERT-base".to_string(),
            ModernBertSize::Large => "answerdotai/ModernBERT-large".to_string(),
        };

        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model));

        let config_filename = repo.get("config.json")?;
        let weights_filename = match repo.get("model.safetensors") {
            Ok(sf) => sf,
            Err(_) => match repo.get("pytorch_model.bin") {
                Ok(pb) => pb,
                Err(e) => {
                    anyhow::bail!(
                        "Model weights not found in repo {}. Expected `model.safetensors` or `pytorch_model.bin`. Error: {e}",
                        model_id
                    )
                }
            },
        };

        let config_content = std::fs::read_to_string(&config_filename).map_err(|e| {
            E::msg(format!(
                "Failed to read config file {config_filename:?}: {e}"
            ))
        })?;
        let config: Config = serde_json::from_str(&config_content)
            .map_err(|e| E::msg(format!("Failed to parse model config: {e}")))?;

        let dtype = DType::F32;
        let vb = if weights_filename
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)? }
        } else if weights_filename.extension().is_some_and(|ext| ext == "bin") {
            VarBuilder::from_pth(&weights_filename, dtype, &device)?
        } else {
            anyhow::bail!("Unsupported weight file format: {:?}", weights_filename);
        };

        let model = ModernBertForMaskedLM::load(vb, &config)?;

        Ok(Self { model, device })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn predict(&self, tokenizer: &Tokenizer, text: &str) -> AnyhowResult<String> {
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;
        let mask_id = tokenizer.token_to_id("[MASK]").unwrap_or(103);
        let mask_index = encoding
            .get_ids()
            .iter()
            .position(|&id| id == mask_id)
            .ok_or_else(|| E::msg("No [MASK] token found in input"))?;

        // Use the tokenizer-provided attention mask instead of a hard-coded one-vector so that
        // padding and special tokens are handled correctly.
        let attention_mask_vals = encoding.get_attention_mask();

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(attention_mask_vals, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let logits = logits.squeeze(0)?.i((mask_index, ..))?;
        let probs = softmax(&logits, D::Minus1)?;
        let predicted = probs.argmax(D::Minus1)?.to_scalar::<u32>()?;
        let token_str = tokenizer
            .decode(&[predicted], true)
            .unwrap_or_default()
            .trim()
            .to_string();
        Ok(text.replace("[MASK]", &token_str))
    }

    pub fn get_tokenizer_repo_info(size: ModernBertSize) -> String {
        match size {
            ModernBertSize::Base => "answerdotai/ModernBERT-base".to_string(),
            ModernBertSize::Large => "answerdotai/ModernBERT-large".to_string(),
        }
    }

    pub fn get_tokenizer(size: ModernBertSize) -> AnyhowResult<Tokenizer> {
        let repo_id = Self::get_tokenizer_repo_info(size);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(repo_id, RepoType::Model));
        let tokenizer_filename = repo.get("tokenizer.json")?;

        Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
    }
}

impl crate::pipelines::fill_mask_pipeline::model::FillMaskModel for FillMaskModernBertModel {
    type Options = ModernBertSize;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        FillMaskModernBertModel::new(options, device)
    }

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> AnyhowResult<String> {
        self.predict(tokenizer, text)
    }

    fn predict_top_k(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        k: usize,
    ) -> AnyhowResult<Vec<crate::pipelines::fill_mask_pipeline::pipeline::FillMaskPrediction>> {
        use crate::pipelines::fill_mask_pipeline::pipeline::FillMaskPrediction;

        if k == 0 {
            return Ok(vec![]);
        }

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;
        let mask_id = tokenizer.token_to_id("[MASK]").unwrap_or(103);
        let mask_index = encoding
            .get_ids()
            .iter()
            .position(|&id| id == mask_id)
            .ok_or_else(|| E::msg("No [MASK] token found in input"))?;

        let attention_mask_vals = encoding.get_attention_mask();

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(attention_mask_vals, &self.device)?.unsqueeze(0)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let logits = logits.squeeze(0)?.i((mask_index, ..))?;
        let probs = softmax(&logits, D::Minus1)?;

        let probs_vec = probs.to_vec1::<f32>()?;
        if probs_vec.is_empty() {
            return Ok(vec![]);
        }

        let mut idxs: Vec<usize> = (0..probs_vec.len()).collect();
        idxs.sort_by(|&i, &j| probs_vec[j].total_cmp(&probs_vec[i]));
        idxs.truncate(std::cmp::min(k, idxs.len()));

        let mut out = Vec::with_capacity(idxs.len());
        for idx in idxs {
            let token_id = idx as u32;
            let token_str = tokenizer
                .decode(&[token_id], true)
                .unwrap_or_default()
                .trim()
                .to_string();
            if token_str.is_empty() {
                continue;
            }
            out.push(FillMaskPrediction {
                word: token_str,
                score: probs_vec[idx],
            });
        }

        Ok(out)
    }

    fn get_tokenizer(options: Self::Options) -> AnyhowResult<Tokenizer> {
        Self::get_tokenizer(options)
    }

    fn device(&self) -> &Device {
        self.device()
    }
}

/// Zero-shot classification model using ModernBERT
#[derive(Clone)]
pub struct ZeroShotModernBertModel {
    model: ModernBertForSequenceClassification,
    device: Device,
    label2id: HashMap<String, u32>,
}

impl ZeroShotModernBertModel {
    pub fn new(size: ModernBertSize, device: Device) -> AnyhowResult<Self> {
        let model_id = match size {
            ModernBertSize::Base => "MoritzLaurer/ModernBERT-base-zeroshot-v2.0".to_string(),
            ModernBertSize::Large => "MoritzLaurer/ModernBERT-large-zeroshot-v2.0".to_string(),
        };

        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model));

        let config_filename = repo.get("config.json")?;
        let weights_filename = {
            match repo.get("model.safetensors") {
                Ok(safetensors) => safetensors,
                Err(_) => match repo.get("pytorch_model.bin") {
                    Ok(pytorch_model) => pytorch_model,
                    Err(e) => {
                        anyhow::bail!("Model weights not found in repo {}. Expected `model.safetensors` or `pytorch_model.bin`. Error: {e}", model_id)
                    }
                },
            }
        };

        let config_content = std::fs::read_to_string(&config_filename).map_err(|e| {
            E::msg(format!(
                "Failed to read config file {config_filename:?}: {e}"
            ))
        })?;

        // Extract classification metadata from config
        #[derive(serde::Deserialize)]
        struct ClassifierConfigRaw {
            id2label: HashMap<String, String>,
            label2id: HashMap<String, u32>,
            classifier_pooling: Option<ClassifierPooling>,
        }
        let class_cfg: ClassifierConfigRaw = serde_json::from_str(&config_content)
            .map_err(|e| E::msg(format!("Failed to parse classifier config: {e}")))?;
        let id2label = class_cfg.id2label;
        let label2id = class_cfg.label2id;
        let classifier_pooling = class_cfg
            .classifier_pooling
            .unwrap_or(ClassifierPooling::MEAN);

        // Parse full model config
        let mut config: Config = serde_json::from_str(&config_content)
            .map_err(|e| E::msg(format!("Failed to parse model config: {e}")))?;

        // Convert label2id values to strings for ClassifierConfig
        let label2id_for_config: HashMap<String, String> = label2id
            .iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect();

        config.classifier_config = Some(ClassifierConfig {
            id2label: id2label.clone(),
            label2id: label2id_for_config,
            classifier_pooling,
        });

        let dtype = DType::F32;
        let vb = if weights_filename
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)? }
        } else if weights_filename.extension().is_some_and(|ext| ext == "bin") {
            VarBuilder::from_pth(&weights_filename, dtype, &device)?
        } else {
            anyhow::bail!("Unsupported weight file format: {:?}", weights_filename);
        };

        let model = ModernBertForSequenceClassification::load(vb, &config)?;

        Ok(Self {
            model,
            device,
            label2id,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn predict(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        self.predict_single_label(tokenizer, text, candidate_labels)
    }

    /// Predict with normalized probabilities for single-label classification (probabilities sum to 1)
    pub fn predict_single_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        let mut results = self.predict_raw(tokenizer, text, candidate_labels)?;

        // Normalize probabilities to sum to 1
        let sum: f32 = results.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in results.iter_mut() {
                *p /= sum;
            }
        }

        Ok(results)
    }

    /// Predict with raw entailment probabilities for multi-label classification
    pub fn predict_multi_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        self.predict_raw(tokenizer, text, candidate_labels)
    }

    /// Core prediction logic that returns raw entailment probabilities
    fn predict_raw(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        if candidate_labels.is_empty() {
            return Ok(vec![]);
        }

        let entailment_id = *self
            .label2id
            .get("entailment")
            .ok_or_else(|| E::msg("Config's label2id map does not contain 'entailment' key"))?;
        let entailment_id_idx = entailment_id as usize;

        let mut encodings = Vec::new();
        for &label in candidate_labels {
            let hypothesis = format!("This example is {label}.");
            let encoding = tokenizer
                .encode((text, hypothesis.as_str()), true)
                .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;
            encodings.push(encoding);
        }

        // Pad the batch
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let pad_token_id = tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0);

        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut all_attention_masks: Vec<u32> = Vec::new();

        for encoding in encodings {
            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = encoding.get_attention_mask().to_vec();

            token_ids.resize(max_len, pad_token_id);
            attention_mask.resize(max_len, 0);

            all_token_ids.extend(token_ids);
            all_attention_masks.extend(attention_mask);
        }

        // Create tensors
        let input_ids_tensor = Tensor::from_vec(
            all_token_ids,
            (candidate_labels.len(), max_len),
            &self.device,
        )?;
        let attention_mask_tensor = Tensor::from_vec(
            all_attention_masks,
            (candidate_labels.len(), max_len),
            &self.device,
        )?;

        // Forward pass
        let logits = self
            .model
            .forward(&input_ids_tensor, &attention_mask_tensor)?;

        // Apply softmax and extract entailment probabilities
        let probabilities = softmax(&logits, D::Minus1)?;
        let entailment_probs = probabilities.i((.., entailment_id_idx))?.to_vec1::<f32>()?;

        // Combine labels and scores
        let mut results: Vec<(String, f32)> = candidate_labels
            .iter()
            .map(|&label| label.to_string())
            .zip(entailment_probs)
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    pub fn get_tokenizer_repo_info(size: ModernBertSize) -> String {
        match size {
            ModernBertSize::Base => "MoritzLaurer/ModernBERT-base-zeroshot-v2.0".to_string(),
            ModernBertSize::Large => "MoritzLaurer/ModernBERT-large-zeroshot-v2.0".to_string(),
        }
    }

    pub fn get_tokenizer(&self, size: ModernBertSize) -> AnyhowResult<Tokenizer> {
        let repo_id = Self::get_tokenizer_repo_info(size);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(repo_id, RepoType::Model));
        let tokenizer_filename = repo.get("tokenizer.json")?;

        Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
    }
}

impl crate::pipelines::zero_shot_classification_pipeline::model::ZeroShotClassificationModel
    for ZeroShotModernBertModel
{
    type Options = ModernBertSize;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        ZeroShotModernBertModel::new(options, device)
    }

    fn predict(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        self.predict_single_label(tokenizer, text, candidate_labels)
    }

    fn predict_multi_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        self.predict_raw(tokenizer, text, candidate_labels)
    }

    fn get_tokenizer(options: Self::Options) -> AnyhowResult<Tokenizer> {
        let repo_id = Self::get_tokenizer_repo_info(options);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(repo_id, RepoType::Model));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
    }

    fn device(&self) -> &Device {
        self.device()
    }
}

/// Sentiment analysis model using ModernBERT
#[derive(Clone)]
pub struct SentimentModernBertModel {
    model: ModernBertForSequenceClassification,
    device: Device,
    id2label: HashMap<String, String>,
}

impl SentimentModernBertModel {
    pub fn new(size: ModernBertSize, device: Device) -> AnyhowResult<Self> {
        let model_id = match size {
            ModernBertSize::Base => "clapAI/modernBERT-base-multilingual-sentiment".to_string(),
            ModernBertSize::Large => "clapAI/modernBERT-large-multilingual-sentiment".to_string(),
        };

        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id, RepoType::Model));

        let config_filename = repo.get("config.json")?;
        let weights_filename = {
            match repo.get("model.safetensors") {
                Ok(safetensors) => safetensors,
                Err(_) => match repo.get("pytorch_model.bin") {
                    Ok(pytorch_model) => pytorch_model,
                    Err(e) => {
                        anyhow::bail!("Model weights not found in repo. Expected `model.safetensors` or `pytorch_model.bin`. Error: {e}")
                    }
                },
            }
        };

        let config_content = std::fs::read_to_string(&config_filename).map_err(|e| {
            E::msg(format!(
                "Failed to read config file {config_filename:?}: {e}"
            ))
        })?;

        // Extract classification metadata
        #[derive(serde::Deserialize)]
        struct ClassifierConfigRaw {
            id2label: HashMap<String, String>,
        }
        let class_cfg: ClassifierConfigRaw = serde_json::from_str(&config_content)
            .map_err(|e| E::msg(format!("Failed to parse classifier config: {e}")))?;
        let id2label = class_cfg.id2label;

        // Parse full model config
        let mut config: Config = serde_json::from_str(&config_content)
            .map_err(|e| E::msg(format!("Failed to parse model config: {e}")))?;

        let label2id = id2label
            .iter()
            .map(|(id, label)| (label.clone(), id.clone()))
            .collect();
        let pooling = config
            .classifier_config
            .as_ref()
            .map(|c| c.classifier_pooling)
            .unwrap_or(ClassifierPooling::MEAN);

        config.classifier_config = Some(ClassifierConfig {
            id2label: id2label.clone(),
            label2id,
            classifier_pooling: pooling,
        });

        let dtype = DType::F32;
        let vb = if weights_filename
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)? }
        } else if weights_filename.extension().is_some_and(|ext| ext == "bin") {
            VarBuilder::from_pth(&weights_filename, dtype, &device)?
        } else {
            anyhow::bail!("Unsupported weight file format: {:?}", weights_filename);
        };

        let model = ModernBertForSequenceClassification::load(vb, &config)?;

        Ok(Self {
            model,
            device,
            id2label,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn predict(&self, tokenizer: &Tokenizer, text: &str) -> AnyhowResult<String> {
        // Tokenize
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;
        let token_ids = tokens.get_ids();
        let attention_mask_vals = tokens.get_attention_mask();

        // Prepare tensors
        let input_ids_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let attention_mask_tensor = Tensor::new(attention_mask_vals, &self.device)?.unsqueeze(0)?;

        // Forward pass
        let output_logits = self
            .model
            .forward(&input_ids_tensor, &attention_mask_tensor)?;

        // Get prediction
        let predictions = output_logits
            .argmax(D::Minus1)?
            .squeeze(0)?
            .to_scalar::<u32>()?;
        let predicted_label = self
            .id2label
            .get(&predictions.to_string())
            .ok_or_else(|| {
                E::msg(format!(
                    "Predicted ID '{predictions}' not found in id2label map"
                ))
            })?
            .clone();

        Ok(predicted_label)
    }

    pub fn get_tokenizer_repo_info(size: ModernBertSize) -> String {
        match size {
            ModernBertSize::Base => "clapAI/modernBERT-base-multilingual-sentiment".to_string(),
            ModernBertSize::Large => "clapAI/modernBERT-large-multilingual-sentiment".to_string(),
        }
    }

    pub fn get_tokenizer(&self, size: ModernBertSize) -> AnyhowResult<Tokenizer> {
        let repo_id = Self::get_tokenizer_repo_info(size);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(repo_id, RepoType::Model));
        let tokenizer_filename = repo.get("tokenizer.json")?;

        Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
    }
}

impl crate::pipelines::sentiment_analysis_pipeline::model::SentimentAnalysisModel
    for SentimentModernBertModel
{
    type Options = ModernBertSize;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        SentimentModernBertModel::new(options, device)
    }

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> AnyhowResult<String> {
        self.predict(tokenizer, text)
    }

    fn predict_with_score(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
    ) -> AnyhowResult<crate::pipelines::sentiment_analysis_pipeline::pipeline::SentimentResult>
    {
        use crate::pipelines::sentiment_analysis_pipeline::pipeline::SentimentResult;

        // Tokenize
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;
        let token_ids = tokens.get_ids();
        let attention_mask_vals = tokens.get_attention_mask();

        // Prepare tensors
        let input_ids_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let attention_mask_tensor = Tensor::new(attention_mask_vals, &self.device)?.unsqueeze(0)?;

        // Forward pass
        let output_logits = self
            .model
            .forward(&input_ids_tensor, &attention_mask_tensor)?;

        // Predicted class id
        let pred_id = output_logits
            .argmax(D::Minus1)?
            .squeeze(0)?
            .to_scalar::<u32>()?;

        // Confidence via softmax
        let probs = softmax(&output_logits, D::Minus1)?;
        let probs_vec = probs.squeeze(0)?.to_vec1::<f32>()?;
        let score = probs_vec.get(pred_id as usize).copied().unwrap_or(0.0);

        let label = self
            .id2label
            .get(&pred_id.to_string())
            .ok_or_else(|| {
                E::msg(format!(
                    "Predicted ID '{pred_id}' not found in id2label map"
                ))
            })?
            .clone();

        Ok(SentimentResult { label, score })
    }

    fn get_tokenizer(options: Self::Options) -> AnyhowResult<Tokenizer> {
        let repo_id = Self::get_tokenizer_repo_info(options);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(repo_id, RepoType::Model));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
    }

    fn device(&self) -> &Device {
        self.device()
    }
}
