use super::model::ZeroShotClassificationModel;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
}

pub struct ZeroShotClassificationPipeline<M: ZeroShotClassificationModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipeline<M> {
    /// Classify with normalized probabilities for single-label classification (probabilities sum to 1)
    pub fn classify(
        &self,
        text: &str,
        candidate_labels: &[&str],
    ) -> anyhow::Result<Vec<ClassificationResult>> {
        let results = self
            .model
            .predict(&self.tokenizer, text, candidate_labels)?;
        Ok(results
            .into_iter()
            .map(|(label, score)| ClassificationResult { label, score })
            .collect())
    }

    /// Classify with raw entailment probabilities for multi-label classification
    pub fn classify_multi_label(
        &self,
        text: &str,
        candidate_labels: &[&str],
    ) -> anyhow::Result<Vec<ClassificationResult>> {
        let results = self
            .model
            .predict_multi_label(&self.tokenizer, text, candidate_labels)?;
        Ok(results
            .into_iter()
            .map(|(label, score)| ClassificationResult { label, score })
            .collect())
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
