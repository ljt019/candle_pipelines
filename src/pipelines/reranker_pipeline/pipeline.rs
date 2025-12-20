use super::model::RerankModel;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Result of reranking a single document.
#[derive(Debug, Clone, Copy)]
pub struct RerankResult {
    /// Index of the document in the input list.
    pub index: usize,
    /// Relevance score as predicted by the model.
    pub score: f32,
}

pub struct RerankPipeline<M: RerankModel> {
    pub(crate) model: Arc<M>,
    pub(crate) tokenizer: Tokenizer,
}

impl<M> RerankPipeline<M>
where
    M: RerankModel + Send + Sync + 'static,
{
    /// Rerank a list of documents against a query.
    /// Returns a list of (document_index, relevance_score) pairs sorted by relevance.
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
    ) -> anyhow::Result<Vec<RerankResult>> {
        let model = Arc::clone(&self.model);
        let tokenizer = self.tokenizer.clone();
        let query_owned = query.to_owned();
        let docs: Vec<String> = documents.iter().map(|d| d.to_string()).collect();
        tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = docs.iter().map(|d| d.as_str()).collect();
            model.rerank(&tokenizer, &query_owned, &refs)
        })
        .await
        .map_err(|e| anyhow::anyhow!(e))?
    }

    /// Batch reranking for multiple queries against the same set of documents.
    pub async fn batch_rerank(
        &self,
        queries: &[&str],
        documents: &[&str],
    ) -> anyhow::Result<Vec<Vec<RerankResult>>> {
        let model = Arc::clone(&self.model);
        let tokenizer = self.tokenizer.clone();
        let queries_owned: Vec<String> = queries.iter().map(|q| q.to_string()).collect();
        let docs_owned: Vec<String> = documents.iter().map(|d| d.to_string()).collect();
        tokio::task::spawn_blocking(move || {
            let q_refs: Vec<&str> = queries_owned.iter().map(|q| q.as_str()).collect();
            let d_refs: Vec<&str> = docs_owned.iter().map(|d| d.as_str()).collect();
            model.batch_rerank(&tokenizer, &q_refs, &d_refs)
        })
        .await
        .map_err(|e| anyhow::anyhow!(e))?
    }

    /// Get the top-k most relevant documents for a query.
    pub async fn rerank_top_k(
        &self,
        query: &str,
        documents: &[&str],
        k: usize,
    ) -> anyhow::Result<Vec<RerankResult>> {
        let mut ranked = self.rerank(query, documents).await?;
        ranked.truncate(k);
        Ok(ranked)
    }

    /// Get relevance scores for all documents without sorting.
    pub async fn get_scores(&self, query: &str, documents: &[&str]) -> anyhow::Result<Vec<f32>> {
        let model = Arc::clone(&self.model);
        let tokenizer = self.tokenizer.clone();
        let query_owned = query.to_owned();
        let docs: Vec<String> = documents.iter().map(|d| d.to_string()).collect();
        let ranked = tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = docs.iter().map(|d| d.as_str()).collect();
            model.rerank(&tokenizer, &query_owned, &refs)
        })
        .await
        .map_err(|e| anyhow::anyhow!(e))??;

        let mut scores = vec![0.0f32; documents.len()];
        for r in ranked {
            scores[r.index] = r.score;
        }
        Ok(scores)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
