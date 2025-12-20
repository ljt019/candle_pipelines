use super::pipeline::RerankResult;
use candle_core::Device;
use tokenizers::Tokenizer;

/// Trait for reranking models.
pub trait RerankModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn rerank(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        documents: &[&str],
    ) -> anyhow::Result<Vec<RerankResult>>;

    fn batch_rerank(
        &self,
        tokenizer: &Tokenizer,
        queries: &[&str],
        documents: &[&str],
    ) -> anyhow::Result<Vec<Vec<RerankResult>>> {
        let mut results = Vec::new();
        for query in queries {
            let ranked = self.rerank(tokenizer, query, documents)?;
            results.push(ranked);
        }
        Ok(results)
    }

    fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer>;

    fn device(&self) -> &Device;
}
