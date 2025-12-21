use tokenizers::Tokenizer;

pub trait FillMaskModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options, device: candle_core::Device) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<String>;

    /// Predict for a batch of inputs, returning a result per item.
    fn predict_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
    ) -> anyhow::Result<Vec<anyhow::Result<String>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict(tokenizer, text))
            .collect())
    }

    /// Return the top-k token predictions for the first `[MASK]` in `text`.
    ///
    /// Default implementation falls back to `predict` (single best) and assigns a score of 1.0.
    fn predict_top_k(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        k: usize,
    ) -> anyhow::Result<Vec<super::pipeline::FillMaskPrediction>> {
        if k == 0 {
            return Ok(vec![]);
        }
        let filled = self.predict(tokenizer, text)?;
        Ok(vec![super::pipeline::FillMaskPrediction {
            word: filled.trim().to_string(),
            score: 1.0,
        }])
    }

    /// Return top-k predictions for a batch of inputs, preserving per-item errors.
    fn predict_top_k_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        k: usize,
    ) -> anyhow::Result<Vec<anyhow::Result<Vec<super::pipeline::FillMaskPrediction>>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict_top_k(tokenizer, text, k))
            .collect())
    }

    fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer>;

    fn device(&self) -> &candle_core::Device;
}
