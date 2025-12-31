#![allow(private_bounds)]

use super::params::{GenerationOverrides, GenerationParams};
use super::pipeline::TextGenerationPipeline;
use super::tools::ErrorStrategy;
use crate::error::Result;
use crate::models::capabilities::{ModelConfig, TextGenerationModel};
use crate::models::{Gemma3, Llama3_2, Olmo3, Qwen3};
use crate::pipelines::cache::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest};

crate::pipelines::utils::impl_device_methods!(direct: TextGenerationPipelineBuilder<C: ModelConfig>);

/// Builder for constructing [`TextGenerationPipeline`] instances.
///
/// # Example
///
/// ```rust,no_run
/// use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3};
///
/// # fn example() -> candle_pipelines::error::Result<()> {
/// // Use .build() for sync or .build_async() for async model loading
/// let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3::Size0_6B)
///     .temperature(0.7)
///     .top_p(0.9)
///     .max_len(512)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
pub struct TextGenerationPipelineBuilder<C: ModelConfig> {
    config: C,
    overrides: GenerationOverrides,
    device_request: DeviceRequest,
    tool_error_strategy: ErrorStrategy,
}

impl<C: ModelConfig> TextGenerationPipelineBuilder<C> {
    /// Create a builder with the given model configuration.
    pub fn new(config: C) -> Self {
        Self {
            config,
            overrides: GenerationOverrides::default(),
            device_request: DeviceRequest::Cpu,
            tool_error_strategy: ErrorStrategy::default(),
        }
    }

    /// Set how tool execution errors are handled.
    pub fn tool_error_strategy(mut self, strategy: ErrorStrategy) -> Self {
        self.tool_error_strategy = strategy;
        self
    }

    /// Set sampling temperature. 0.0 = deterministic, higher = more random.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.overrides.temperature = Some(temperature);
        self
    }

    /// Set penalty for repeating tokens. 1.0 = no penalty.
    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.overrides.repeat_penalty = Some(repeat_penalty);
        self
    }

    /// Alias for [`repeat_penalty`](Self::repeat_penalty).
    pub fn repetition_penalty(self, repetition_penalty: f32) -> Self {
        self.repeat_penalty(repetition_penalty)
    }

    /// Set how many recent tokens to consider for repeat penalty.
    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.overrides.repeat_last_n = Some(repeat_last_n);
        self
    }

    /// Set random seed for reproducible generation.
    pub fn seed(mut self, seed: u64) -> Self {
        self.overrides.seed = Some(seed);
        self
    }

    /// Set maximum tokens to generate per turn.
    pub fn max_len(mut self, max_len: usize) -> Self {
        self.overrides.max_len = Some(max_len);
        self
    }

    /// Alias for [`max_len`](Self::max_len).
    pub fn max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.overrides.max_len = Some(max_new_tokens);
        self
    }

    /// Set nucleus sampling threshold (0.0-1.0).
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.overrides.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    /// Only sample from the top k most likely tokens.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.overrides.top_k = Some(top_k);
        self
    }

    /// Filter tokens below min_p * max_probability (0.0-1.0).
    pub fn min_p(mut self, min_p: f64) -> Self {
        self.overrides.min_p = Some(min_p.clamp(0.0, 1.0));
        self
    }

    /// Build the pipeline synchronously, downloading and loading the model if needed.
    ///
    /// Uses `ureq` for HTTP requests. For async builds with `reqwest`, use [`build_async`](Self::build_async).
    pub fn build(self) -> Result<TextGenerationPipeline<C>>
    where
        C: ModelOptions + 'static,
        C::Model: TextGenerationModel,
    {
        let device = self.device_request.resolve()?;
        let cache_key = build_cache_key(&self.config, &device);

        let config = self.config;
        let device_for_model = device.clone();
        let model = global_cache().get_or_create(&cache_key, || config.build(device_for_model))?;

        let gen_params = GenerationParams::resolve(model.get_generation_config(), &self.overrides)?;

        TextGenerationPipeline::new(model, gen_params, device, self.tool_error_strategy)
    }

    /// Build the pipeline asynchronously, downloading and loading the model if needed.
    ///
    /// Uses `reqwest` for HTTP requests. For sync builds with `ureq`, use [`build`](Self::build).
    pub async fn build_async(self) -> Result<TextGenerationPipeline<C>>
    where
        C: ModelOptions + 'static,
        C::Model: TextGenerationModel,
    {
        let device = self.device_request.resolve()?;
        let cache_key = build_cache_key(&self.config, &device);

        let config = self.config;
        let device_for_model = device.clone();
        let model = global_cache()
            .get_or_create_async(&cache_key, || async move { config.build(device_for_model) })
            .await?;

        let gen_params = GenerationParams::resolve(model.get_generation_config(), &self.overrides)?;

        TextGenerationPipeline::new(model, gen_params, device, self.tool_error_strategy)
    }
}

impl TextGenerationPipelineBuilder<Qwen3> {
    /// Create a builder for a Qwen 3 model.
    pub fn qwen3(config: Qwen3) -> Self {
        Self::new(config)
    }
}

impl TextGenerationPipelineBuilder<Gemma3> {
    /// Create a builder for a Gemma 3 model.
    pub fn gemma3(config: Gemma3) -> Self {
        Self::new(config)
    }
}

impl TextGenerationPipelineBuilder<Llama3_2> {
    /// Create a builder for a Llama 3.2 model.
    pub fn llama3_2(config: Llama3_2) -> Self {
        Self::new(config)
    }
}

impl TextGenerationPipelineBuilder<Olmo3> {
    /// Create a builder for an OLMo-3 model.
    pub fn olmo3(config: Olmo3) -> Self {
        Self::new(config)
    }
}
