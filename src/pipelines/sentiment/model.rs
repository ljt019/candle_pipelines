use tokenizers::Tokenizer;

pub trait SentimentAnalysisModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options, device: candle_core::Device) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<String>;

    /// Predict sentiment and return both label + confidence score.
    ///
    /// Default implementation falls back to `predict` and assigns a score of 1.0.
    fn predict_with_score(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
    ) -> anyhow::Result<super::pipeline::SentimentResult> {
        let label = self.predict(tokenizer, text)?;
        Ok(super::pipeline::SentimentResult { label, score: 1.0 })
    }

    fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer>;

    fn device(&self) -> &candle_core::Device;
}
