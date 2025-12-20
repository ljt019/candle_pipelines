use super::model::EmbeddingModel;
use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct EmbeddingPipeline<M: EmbeddingModel> {
    pub(crate) model: Arc<M>,
    pub(crate) tokenizer: Tokenizer,
}

impl<M> EmbeddingPipeline<M>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    /// Compute the embedding for a single text asynchronously.
    pub async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let model = Arc::clone(&self.model);
        let tokenizer = self.tokenizer.clone();
        let text = text.to_owned();
        tokio::task::spawn_blocking(move || model.embed(&tokenizer, &text))
            .await
            .map_err(|e| anyhow::anyhow!(e))?
    }

    /// Compute embeddings for a batch of texts asynchronously.
    pub async fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let model = Arc::clone(&self.model);
        let tokenizer = self.tokenizer.clone();
        let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
            model.embed_batch(&tokenizer, &refs)
        })
        .await
        .map_err(|e| anyhow::anyhow!(e))?
    }

    /// Calculate cosine similarity between two embeddings.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }

    /// Return the top-k most similar embeddings to the query embedding.
    pub fn top_k(query: &[f32], embeddings: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, Self::cosine_similarity(query, emb)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(std::cmp::min(k, scored.len()));
        scored
    }

    /// Convenience wrapper that embeds the query text and returns the top-k most
    /// similar precomputed embeddings.
    pub async fn recall_top_k(
        &self,
        query: &str,
        embeddings: &[Vec<f32>],
        k: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let query_emb = self.embed(query).await?;
        Ok(Self::top_k(&query_emb, embeddings, k))
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
