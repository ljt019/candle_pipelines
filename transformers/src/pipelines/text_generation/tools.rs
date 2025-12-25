use crate::error::Result;
use futures::future::BoxFuture;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub enum ErrorStrategy {
    #[default]
    Fail,
    ReturnToModel,
}

pub trait ToolCalling {
    fn register_tool(&mut self, tool: Tool);
    fn unregister_tool(&mut self, name: &str);
    fn clear_tools(&mut self);
    fn registered_tools(&self) -> Vec<Tool>;
}

pub type ToolFuture = BoxFuture<'static, Result<String>>;

#[derive(serde::Serialize)]
pub struct Tool {
    pub(crate) name: String,
    pub(crate) description: String,
    #[serde(rename = "parameters")]
    pub(crate) schema: schemars::schema::RootSchema,
    #[serde(skip_serializing)]
    pub(crate) function: Arc<dyn Fn(serde_json::Value) -> ToolFuture + Send + Sync>,
    #[serde(skip_serializing)]
    pub(crate) max_retries: u32,
}

impl Clone for Tool {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema: self.schema.clone(),
            function: Arc::clone(&self.function),
            max_retries: self.max_retries,
        }
    }
}

impl Tool {
    /// Creates a new Tool. This is primarily used by the `#[tool]` macro.
    #[doc(hidden)]
    pub fn new(
        name: String,
        description: String,
        schema: schemars::schema::RootSchema,
        function: impl Fn(serde_json::Value) -> ToolFuture + Send + Sync + 'static,
        max_retries: u32,
    ) -> Self {
        Self {
            name,
            description,
            schema,
            function: Arc::new(function),
            max_retries,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub(crate) async fn call(&self, parameters: serde_json::Value) -> Result<String> {
        self.validate(&parameters)?;
        (self.function)(parameters).await
    }

    pub fn schema(&self) -> &schemars::schema::RootSchema {
        &self.schema
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub(crate) fn max_retries(&self) -> u32 {
        self.max_retries
    }

    fn validate(&self, params: &serde_json::Value) -> Result<()> {
        let schema = serde_json::to_value(&self.schema).map_err(|e| {
            crate::error::TransformersError::Tool(
                format!("Schema error for '{}': schema serialization failed: {}", self.name, e)
            )
        })?;
        let compiled = jsonschema::JSONSchema::options()
            .with_draft(jsonschema::Draft::Draft7)
            .compile(&schema)
            .map_err(|e| {
                crate::error::TransformersError::Tool(
                    format!("Schema error for '{}': invalid schema: {}", self.name, e)
                )
            })?;

        let validation_result = compiled.validate(params);
        match validation_result {
            Ok(_) => Ok(()),
            Err(errors) => {
                let messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                let error_msg = messages.join(", ");
                Err(crate::error::TransformersError::Tool(
                    format!("Invalid parameters for '{}': {}", self.name, error_msg)
                ))
            }
        }
    }
}
