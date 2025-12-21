use super::model::SentimentAnalysisModel;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub label: String,
    pub score: f32,
}

pub struct SentimentAnalysisPipeline<M: SentimentAnalysisModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: SentimentAnalysisModel> SentimentAnalysisPipeline<M> {
    /// Predict sentiment with structured result containing label and confidence score
    pub fn predict(&self, text: &str) -> anyhow::Result<SentimentResult> {
        self.model.predict_with_score(&self.tokenizer, text)
    }

    /// Predict sentiment for a batch of inputs.
    pub fn predict_batch(
        &self,
        texts: &[&str],
    ) -> anyhow::Result<Vec<anyhow::Result<SentimentResult>>> {
        self.model.predict_with_score_batch(&self.tokenizer, texts)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
