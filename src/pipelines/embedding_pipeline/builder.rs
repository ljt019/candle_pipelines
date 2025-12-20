use super::model::EmbeddingModel;
use super::pipeline::EmbeddingPipeline;
use crate::core::ModelOptions;
use crate::pipelines::utils::{
    BasePipelineBuilder, DeviceRequest, DeviceSelectable, StandardPipelineBuilder,
};
use std::sync::Arc;

pub struct EmbeddingPipelineBuilder<M: EmbeddingModel>(StandardPipelineBuilder<M::Options>);

impl<M: EmbeddingModel> EmbeddingPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self(StandardPipelineBuilder::new(options))
    }
}

impl<M: EmbeddingModel> DeviceSelectable for EmbeddingPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        self.0.device_request_mut()
    }
}

impl<M: EmbeddingModel> BasePipelineBuilder<M> for EmbeddingPipelineBuilder<M>
where
    M: Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = EmbeddingPipeline<M>;
    type Options = M::Options;

    fn options(&self) -> &Self::Options {
        &self.0.options
    }

    fn device_request(&self) -> &DeviceRequest {
        &self.0.device_request
    }

    fn create_model(options: Self::Options, device: candle_core::Device) -> anyhow::Result<M> {
        M::new(options, device)
    }

    fn get_tokenizer(options: Self::Options) -> anyhow::Result<tokenizers::Tokenizer> {
        M::get_tokenizer(options)
    }

    fn construct_pipeline(
        model: M,
        tokenizer: tokenizers::Tokenizer,
    ) -> anyhow::Result<Self::Pipeline> {
        Ok(EmbeddingPipeline {
            model: Arc::new(model),
            tokenizer,
        })
    }
}

impl
    EmbeddingPipelineBuilder<crate::models::implementations::qwen3_embeddings::Qwen3EmbeddingModel>
{
    pub fn qwen3(
        size: crate::models::implementations::qwen3_embeddings::Qwen3EmbeddingSize,
    ) -> Self {
        Self::new(size)
    }
}
