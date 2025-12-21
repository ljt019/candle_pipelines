extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Attribute, Expr, FnArg, ItemFn, Lit, Meta, Pat, ReturnType, Type};

/// Extract the doc comments on the original function, concatenated and trimmed.
fn extract_doc(attrs: &[Attribute]) -> String {
    let mut out = String::new();
    for attr in attrs {
        if let Meta::NameValue(nv) = &attr.meta {
            if nv.path.is_ident("doc") {
                if let Expr::Lit(expr_lit) = &nv.value {
                    if let Lit::Str(lit_str) = &expr_lit.lit {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(lit_str.value().trim());
                    }
                }
            }
        }
    }
    out
}

/// Parse the tool attribute arguments for error strategy and retries.
fn parse_tool_config(args: TokenStream) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    let default_error_strategy =
        quote! { transformers::pipelines::text_generation::ErrorStrategy::Fail };
    let default_retries = quote! { 3u32 };

    if args.is_empty() {
        return (default_error_strategy, default_retries);
    }

    let mut error_strategy = default_error_strategy;
    let mut retries = default_retries;

    // Try to parse as multiple comma-separated arguments
    let args_str = args.to_string();

    // Split by comma and process each part
    for part in args_str.split(',') {
        let part = part.trim();

        if part.starts_with("on_error") {
            // Extract the value after the =
            if let Some(value_part) = part.split('=').nth(1) {
                let value_part = value_part.trim();
                if let Ok(expr) = syn::parse_str::<syn::Expr>(value_part) {
                    error_strategy = parse_error_strategy_from_expr(&expr);
                }
            }
        } else if part.starts_with("retries") {
            // Extract the value after the =
            if let Some(value_part) = part.split('=').nth(1) {
                let value_part = value_part.trim();
                if let Ok(lit) = syn::parse_str::<syn::LitInt>(value_part) {
                    let retry_count = lit.base10_parse::<u32>().unwrap_or(3);
                    retries = quote! { #retry_count };
                }
            }
        }
    }

    (error_strategy, retries)
}

fn parse_error_strategy_from_expr(expr: &syn::Expr) -> proc_macro2::TokenStream {
    // Convert the expression to a string to check what it contains
    let expr_str = quote!(#expr).to_string();

    // Clean up the string (remove extra spaces)
    let expr_str = expr_str.replace(" ", "");

    // Handle different forms of the error strategy
    if expr_str == "Fail" || expr_str.contains("ErrorStrategy::Fail") {
        quote! { transformers::pipelines::text_generation::ErrorStrategy::Fail }
    } else if expr_str == "ReturnToModel" || expr_str.contains("ErrorStrategy::ReturnToModel") {
        quote! { transformers::pipelines::text_generation::ErrorStrategy::ReturnToModel }
    } else {
        // Generate a compile-time error for invalid strategies
        syn::Error::new_spanned(
            expr,
            "Unknown error strategy. Valid options are: ErrorStrategy::Fail, ErrorStrategy::ReturnToModel"
        ).to_compile_error()
    }
}

/// Check if the function returns a Result type
fn returns_result(output: &ReturnType) -> bool {
    if let ReturnType::Type(_, ty) = output {
        if let Type::Path(type_path) = &**ty {
            if let Some(segment) = type_path.path.segments.last() {
                return segment.ident == "Result";
            }
        }
    }
    false
}

/// Very small type-name mapper.
fn type_to_json_type(ty: &Type) -> String {
    if let Type::Path(tp) = ty {
        if let Some(ident) = tp.path.get_ident() {
            let name = ident.to_string();
            return match name.as_str() {
                "String" | "str" => "string".to_string(),
                "i32" | "i64" | "u32" | "u64" | "f32" | "f64" | "usize" | "isize" => {
                    "number".to_string()
                }
                "bool" => "boolean".to_string(),
                _ => "object".to_string(),
            };
        }
    }
    "object".to_string()
}

/// Attribute macro `#[tool]` that turns a nice Rust function into a `Tool`.
///
/// Example:
/// ```
/// use transformers::tool;
///
/// #[tool]
/// fn add(a: i32, b: i32) -> String {
///     (a + b).to_string()
/// }
///
/// #[tool(on_error = ErrorStrategy::ReturnToModel)]
/// fn get_weather(city: String) -> Result<String, WeatherError> {
///     // ...
/// }
///
/// // In user code:
/// let mut pipeline = ...;
/// pipeline.register_tools(tools![add]);
/// ```
#[proc_macro_attribute]
pub fn tool(args: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the error strategy and retries from args
    let (error_strategy, max_retries) = parse_tool_config(args);

    // Parse the source function.
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name_ident = &input_fn.sig.ident;
    let fn_name_str = fn_name_ident.to_string();
    let wrapper_name = format_ident!("__{}_tool_wrapper", fn_name_ident);
    let tool_builder_name = format_ident!("__{}_tool_builder", fn_name_ident);

    // Doc comments become description.
    let description = extract_doc(&input_fn.attrs);

    // Check if function returns Result
    let is_result = returns_result(&input_fn.sig.output);

    // Gather parameter information.
    let mut schema_kv_pairs = Vec::new();
    let mut extraction_stmts = Vec::new();
    let mut call_args = Vec::new();

    // Traverse function arguments.
    for arg in input_fn.sig.inputs.iter() {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = &pat_ident.ident;
                let param_name_str = param_name.to_string();

                let ty = &*pat_type.ty;
                let json_type_str = type_to_json_type(ty);

                // Build the schema key-value pair
                schema_kv_pairs.push(quote! {
                    parameters.insert(#param_name_str.to_string(), #json_type_str.to_string());
                });

                // Extract the parameter from the HashMap at runtime
                extraction_stmts.push(quote! {
                    let #param_name: #ty = parameters
                        .remove(#param_name_str)
                        .expect("Missing parameter for tool call")
                        .parse()
                        .expect("Failed to parse parameter for tool call");
                });

                // Pass it to the original function
                call_args.push(quote! { #param_name });
            }
        }
    }

    // Generate different wrapper logic based on return type
    let wrapper_body = if is_result {
        quote! {
            #( #extraction_stmts )*
            use transformers::pipelines::text_generation::ToolError;
            let result = #fn_name_ident( #(#call_args),* );

            // Convert the result to the expected type
            match result {
                Ok(s) => Ok(s),
                Err(e) => Err(ToolError::Message(e.to_string())),
            }
        }
    } else {
        quote! {
            #( #extraction_stmts )*
            let result = #fn_name_ident( #(#call_args),* );
            Ok(result)
        }
    };

    // Generate the output tokens: keep original fn, plus wrapper + data.
    let expanded = quote! {
        // Keep the original function as-is
        #input_fn

        // Automatically generated wrapper that matches the `Tool` function signature.
        #[doc(hidden)]
        fn #wrapper_name(mut parameters: std::collections::HashMap<String, String>) -> Result<String, transformers::pipelines::text_generation::ToolError> {
            #wrapper_body
        }

        // Hidden function used by the tools! macro
        #[doc(hidden)]
        pub fn #tool_builder_name() -> transformers::pipelines::text_generation::Tool {
            let mut parameters = std::collections::HashMap::new();
            #( #schema_kv_pairs )*

            transformers::pipelines::text_generation::Tool::new(
                #fn_name_str.to_string(),
                #description.to_string(),
                parameters,
                #wrapper_name,
                #error_strategy,
                #max_retries,
            )
        }

        // Generate a module with the same name as the function
        // This allows the tools! macro to find the builder
        #[doc(hidden)]
        pub mod #fn_name_ident {
            use super::*;

            pub fn __tool() -> transformers::pipelines::text_generation::Tool {
                #tool_builder_name()
            }
        }
    };

    TokenStream::from(expanded)
}
