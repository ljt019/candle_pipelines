use crate::core::{global_cache, ModelOptions};
use crate::models::{Gemma3Model, Gemma3Size, Qwen3Model, Qwen3Size};
use crate::pipelines::utils::{build_cache_key, DeviceRequest, DeviceSelectable};

use super::model::TextGenerationModel;
use super::parser::XmlParserBuilder;
use super::pipeline::TextGenerationPipeline;
use super::xml_pipeline::XmlGenerationPipeline;

/// Builder for text generation pipelines.
///
/// Note: This builder doesn't use the shared `BasePipelineBuilder` trait because
/// it has a more complex building pattern with many configuration options
/// (temperature, top_p, etc.), async model creation, and different caching logic.
/// The shared trait is designed for simpler builders with just options and device.
pub struct TextGenerationPipelineBuilder<M: TextGenerationModel> {
    model_options: M::Options,
    temperature: Option<f64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    seed: Option<u64>,
    max_len: Option<usize>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    min_p: Option<f64>,
    device_request: DeviceRequest,
}

impl<M: TextGenerationModel> TextGenerationPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            model_options: options,
            temperature: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
            max_len: None,
            top_p: None,
            top_k: None,
            min_p: None,
            device_request: DeviceRequest::Default,
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = Some(repeat_penalty);
        self
    }

    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.repeat_last_n = Some(repeat_last_n);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn max_len(mut self, max_len: usize) -> Self {
        self.max_len = Some(max_len);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn min_p(mut self, min_p: f64) -> Self {
        self.min_p = Some(min_p.clamp(0.0, 1.0));
        self
    }

    pub async fn build(self) -> anyhow::Result<TextGenerationPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        // Resolve device first so model weights and inference tensors live on the same device.
        let device = self.device_request.resolve()?;

        // Include device in the cache key so CPU/GPU variants don't get mixed.
        let cache_key = build_cache_key(&self.model_options, &device);

        // Always use the global cache to share models (weights) across pipelines.
        let options = self.model_options.clone();
        let device_for_model = device.clone();
        let model = global_cache()
            .get_or_create_async(&cache_key, || async move {
                M::new(options, device_for_model).await
            })
            .await?;

        // Start with model-specific defaults
        let default_params = model.default_generation_params();

        // Override with any user-specified values
        let gen_params = crate::models::generation::GenerationParams::new(
            self.temperature.unwrap_or(default_params.temperature),
            self.repeat_penalty.unwrap_or(default_params.repeat_penalty),
            self.repeat_last_n.unwrap_or(default_params.repeat_last_n),
            self.seed.unwrap_or_else(rand::random::<u64>),
            self.max_len.unwrap_or(default_params.max_len),
            self.top_p.unwrap_or(default_params.top_p),
            self.top_k.unwrap_or(default_params.top_k),
            self.min_p.unwrap_or(default_params.min_p),
        );
        TextGenerationPipeline::new(model, gen_params, device).await
    }

    pub async fn build_xml(self, tags: &[&str]) -> anyhow::Result<XmlGenerationPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        // Resolve device first so model weights and inference tensors live on the same device.
        let device = self.device_request.resolve()?;

        // Include device in the cache key so CPU/GPU variants don't get mixed.
        let cache_key = build_cache_key(&self.model_options, &device);

        // Always use the global cache to share models (weights) across pipelines.
        let options = self.model_options.clone();
        let device_for_model = device.clone();
        let model = global_cache()
            .get_or_create_async(&cache_key, || async move {
                M::new(options, device_for_model).await
            })
            .await?;

        // Start with model-specific defaults
        let default_params = model.default_generation_params();

        // Override with any user-specified values
        let gen_params = crate::models::generation::GenerationParams::new(
            self.temperature.unwrap_or(default_params.temperature),
            self.repeat_penalty.unwrap_or(default_params.repeat_penalty),
            self.repeat_last_n.unwrap_or(default_params.repeat_last_n),
            self.seed.unwrap_or_else(rand::random::<u64>),
            self.max_len.unwrap_or(default_params.max_len),
            self.top_p.unwrap_or(default_params.top_p),
            self.top_k.unwrap_or(default_params.top_k),
            self.min_p.unwrap_or(default_params.min_p),
        );

        let mut builder = XmlParserBuilder::new();
        for tag in tags {
            builder.register_tag(*tag);
        }
        let xml_parser = builder.build();
        XmlGenerationPipeline::new(model, gen_params, xml_parser, device).await
    }
}

impl<M: TextGenerationModel> DeviceSelectable for TextGenerationPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}

impl TextGenerationPipelineBuilder<Qwen3Model> {
    pub fn qwen3(size: Qwen3Size) -> Self {
        Self::new(size)
    }
}

impl TextGenerationPipelineBuilder<Gemma3Model> {
    pub fn gemma3(size: Gemma3Size) -> Self {
        Self::new(size)
    }
}
