use super::pipeline::ReaderPipeline;
use crate::error::Result;
use crate::models::readerlm_v2::ReaderLM;
use crate::pipelines::text_generation::params::GenerationParams;

/// Builder for creating [`ReaderPipeline`] instances.
///
/// # Example
///
/// ```rust,no_run
/// use candle_pipelines::reader::ReaderPipelineBuilder;
///
/// # async fn example() -> candle_pipelines::error::Result<()> {
/// let pipeline = ReaderPipelineBuilder::new()
///     .temperature(0.65)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct ReaderPipelineBuilder {
    gen_params: GenerationParams,
    device: DeviceRequest,
}

#[derive(Clone, Default)]
enum DeviceRequest {
    #[default]
    Cpu,
    Cuda(usize),
}

impl Default for ReaderPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ReaderPipelineBuilder {
    /// Create a new builder with ReaderLM-v2 default settings.
    ///
    /// Defaults from `generation_config.json`:
    /// - temperature: 0.65
    /// - top_k: 20
    /// - top_p: 0.8
    /// - repeat_penalty: 1.08
    /// - max_len: 8192
    pub fn new() -> Self {
        Self {
            gen_params: GenerationParams {
                temperature: 0.65,
                top_k: Some(20),
                top_p: Some(0.8),
                repeat_penalty: 1.08,
                repeat_last_n: 64,
                seed: rand::random(),
                max_len: 8192,
                min_p: None,
            },
            device: DeviceRequest::Cpu,
        }
    }

    /// Set sampling temperature (default: 0.65).
    ///
    /// 0.0 = deterministic (greedy), higher = more random.
    pub fn temperature(mut self, temp: f64) -> Self {
        self.gen_params.temperature = temp;
        self
    }

    /// Set maximum output tokens (default: 8192).
    pub fn max_output_tokens(mut self, tokens: usize) -> Self {
        self.gen_params.max_len = tokens;
        self
    }

    /// Set top-k sampling (default: 20).
    pub fn top_k(mut self, k: usize) -> Self {
        self.gen_params.top_k = Some(k);
        self
    }

    /// Set nucleus sampling threshold (default: 0.8).
    pub fn top_p(mut self, p: f64) -> Self {
        self.gen_params.top_p = Some(p);
        self
    }

    /// Set repeat penalty (default: 1.08).
    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.gen_params.repeat_penalty = penalty;
        self
    }

    /// Set random seed for reproducible generation.
    pub fn seed(mut self, seed: u64) -> Self {
        self.gen_params.seed = seed;
        self
    }

    /// Run on CPU (default).
    pub fn cpu(mut self) -> Self {
        self.device = DeviceRequest::Cpu;
        self
    }

    /// Run on CUDA GPU.
    ///
    /// Requires the `cuda` feature to be enabled.
    pub fn cuda(mut self, device_id: usize) -> Self {
        self.device = DeviceRequest::Cuda(device_id);
        self
    }

    /// Build the pipeline.
    pub async fn build(self) -> Result<ReaderPipeline<ReaderLM>> {
        let device = match self.device {
            DeviceRequest::Cpu => candle_core::Device::Cpu,
            DeviceRequest::Cuda(id) => candle_core::Device::new_cuda(id)?,
        };

        let model = ReaderLM::from_hf(&device).await?;
        let tokenizer = ReaderLM::get_tokenizer().await?;

        Ok(ReaderPipeline {
            model,
            tokenizer,
            gen_params: self.gen_params,
        })
    }
}
