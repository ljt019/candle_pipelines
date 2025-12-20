//! Model caching utilities for sharing weights across multiple pipelines.
//!
//! This module provides a thread-safe cache for model instances, allowing
//! multiple pipelines to share the same underlying model weights while
//! maintaining independent inference contexts.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Trait implemented by model option types to generate a stable cache key.
pub trait ModelOptions {
    fn cache_key(&self) -> String;
}

/// Type alias for the complex cache storage type.
type CacheStorage = HashMap<(TypeId, String), Arc<dyn Any + Send + Sync>>;

/// A thread-safe cache for model instances.
///
/// The cache stores models by a string key (typically the model size/variant)
/// and ensures that multiple requests for the same model return clones that
/// share the underlying weights.
pub struct ModelCache {
    cache: Arc<Mutex<CacheStorage>>,
}

impl ModelCache {
    /// Create a new empty model cache.
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get or create a model from the cache.
    ///
    /// If a model with the given key already exists, a clone is returned.
    /// Otherwise, the loader function is called to create a new model instance.
    ///
    /// # Arguments
    /// * `key` - A unique identifier for this model variant (e.g., "qwen3-4b")
    /// * `loader` - A function that creates a new model instance if not cached
    ///
    /// # Type Parameters
    /// * `M` - The model type, must be Clone + Send + Sync
    pub async fn get_or_create<M, F>(&self, key: &str, loader: F) -> anyhow::Result<M>
    where
        M: Clone + Send + Sync + 'static,
        F: FnOnce() -> anyhow::Result<M>,
    {
        let type_id = TypeId::of::<M>();
        let cache_key = (type_id, key.to_string());

        // First, try to get from cache
        {
            let cache = self.cache.lock().await;
            if let Some(cached) = cache.get(&cache_key) {
                if let Some(model) = cached.downcast_ref::<M>() {
                    return Ok(model.clone());
                }
            }
        }

        // Not in cache, create new instance
        let model = loader()?;

        // Store in cache
        {
            let mut cache = self.cache.lock().await;
            cache.insert(
                cache_key,
                Arc::new(model.clone()) as Arc<dyn Any + Send + Sync>,
            );
        }

        Ok(model)
    }

    pub async fn get_or_create_async<M, Fut, F>(&self, key: &str, loader: F) -> anyhow::Result<M>
    where
        M: Clone + Send + Sync + 'static,
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<M>>,
    {
        let type_id = TypeId::of::<M>();
        let cache_key = (type_id, key.to_string());

        {
            let cache = self.cache.lock().await;
            if let Some(cached) = cache.get(&cache_key) {
                if let Some(model) = cached.downcast_ref::<M>() {
                    return Ok(model.clone());
                }
            }
        }

        let model = loader().await?;

        {
            let mut cache = self.cache.lock().await;
            cache.insert(
                cache_key,
                Arc::new(model.clone()) as Arc<dyn Any + Send + Sync>,
            );
        }

        Ok(model)
    }

    /// Clear all cached models.
    pub async fn clear(&self) {
        let mut cache = self.cache.lock().await;
        cache.clear();
    }

    /// Get the number of cached models.
    pub async fn len(&self) -> usize {
        let cache = self.cache.lock().await;
        cache.len()
    }

    /// Check if the cache is empty.
    pub async fn is_empty(&self) -> bool {
        let cache = self.cache.lock().await;
        cache.is_empty()
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Global model cache instance.
///
/// This provides a convenient way to share models across the entire application
/// without having to pass the cache around.
static GLOBAL_MODEL_CACHE: once_cell::sync::Lazy<ModelCache> =
    once_cell::sync::Lazy::new(ModelCache::new);

/// Get a reference to the global model cache.
pub fn global_cache() -> &'static ModelCache {
    &GLOBAL_MODEL_CACHE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestModel {
        id: String,
    }

    #[tokio::test]
    async fn test_cache_returns_same_instance() {
        let cache = ModelCache::new();

        let model1 = cache
            .get_or_create::<TestModel, _>("test-model", || {
                Ok(TestModel {
                    id: "original".to_string(),
                })
            })
            .await
            .unwrap();

        let model2 = cache
            .get_or_create::<TestModel, _>("test-model", || {
                // This should not be called
                Ok(TestModel {
                    id: "new".to_string(),
                })
            })
            .await
            .unwrap();

        assert_eq!(model1.id, model2.id);
        assert_eq!(model1.id, "original");
    }

    #[tokio::test]
    async fn test_cache_separates_types_for_same_key() {
        let cache = ModelCache::new();

        #[derive(Clone)]
        struct A(&'static str);
        #[derive(Clone)]
        struct B(&'static str);

        let a = cache
            .get_or_create::<A, _>("same-key", || Ok(A("a")))
            .await
            .unwrap();
        let b = cache
            .get_or_create::<B, _>("same-key", || Ok(B("b")))
            .await
            .unwrap();

        assert_eq!(a.0, "a");
        assert_eq!(b.0, "b");
    }

    #[tokio::test]
    async fn test_cache_clear_empties_storage() {
        let cache = ModelCache::new();
        #[derive(Clone)]
        struct A;
        let _ = cache.get_or_create::<A, _>("k", || Ok(A)).await.unwrap();
        assert!(!cache.is_empty().await);
        cache.clear().await;
        assert!(cache.is_empty().await);
        assert_eq!(cache.len().await, 0);
    }
}
