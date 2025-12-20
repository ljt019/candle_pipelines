use super::model::RerankModel;
use super::pipeline::RerankPipeline;
use crate::core::ModelOptions;
use crate::pipelines::utils::{
    BasePipelineBuilder, DeviceRequest, DeviceSelectable, StandardPipelineBuilder,
};
use std::sync::Arc;

pub struct RerankPipelineBuilder<M: RerankModel>(StandardPipelineBuilder<M::Options>);

impl<M: RerankModel> RerankPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self(StandardPipelineBuilder::new(options))
    }
}

impl<M: RerankModel> DeviceSelectable for RerankPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        self.0.device_request_mut()
    }
}

impl<M: RerankModel> BasePipelineBuilder<M> for RerankPipelineBuilder<M>
where
    M: Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = RerankPipeline<M>;
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
        Ok(RerankPipeline {
            model: Arc::new(model),
            tokenizer,
        })
    }
}

impl RerankPipelineBuilder<crate::models::implementations::qwen3_reranker::Qwen3RerankModel> {
    pub fn qwen3(size: crate::models::implementations::qwen3_reranker::Qwen3RerankSize) -> Self {
        Self::new(size)
    }
}
