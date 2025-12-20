use super::model::FillMaskModel;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct FillMaskPrediction {
    pub word: String,
    pub score: f32,
}

pub struct FillMaskPipeline<M: FillMaskModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: FillMaskModel> FillMaskPipeline<M> {
    /// Return the top prediction for the masked token
    pub fn predict(&self, text: &str) -> anyhow::Result<FillMaskPrediction> {
        let predictions = self.predict_top_k(text, 1)?;
        predictions
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No predictions returned"))
    }

    /// Return top-k predictions with scores for ranking/choice
    pub fn predict_top_k(&self, text: &str, k: usize) -> anyhow::Result<Vec<FillMaskPrediction>> {
        self.model.predict_top_k(&self.tokenizer, text, k)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
