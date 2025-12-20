use tokenizers::Tokenizer;

pub trait FillMaskModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options, device: candle_core::Device) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<String>;

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

    fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer>;

    fn device(&self) -> &candle_core::Device;
}
