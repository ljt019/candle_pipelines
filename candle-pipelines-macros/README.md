# candle-pipelines-macros

[<img alt="crates.io" src="https://img.shields.io/crates/v/candle-pipelines-macros.svg?style=for-the-badge&color=fc8d62&logo=rust" height="19">](https://crates.io/crates/candle-pipelines-macros)

> [!warning]
> ***This crate provides macros used by `candle-pipelines`. It is automatically included as a dependency — you should not need to add it directly.***

Internal procedural macros for [`candle-pipelines`](https://crates.io/crates/candle-pipelines).

## Provided Macros

- `#[tool]` - Convert functions into tools the model can call
- `tools![]` - Collect multiple tools into a `Vec<Tool>`
- `#[derive(XmlTag)]` - Derive XML tag parsing for enums