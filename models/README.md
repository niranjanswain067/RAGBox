# Models Directory

This directory stores the local quantized Large Language Model (GGUF format) used by RAGBox.

## Expected Model File

By default, RAGBox looks for:
```text
models/qwen3-int4.gguf
```

## Supported Format
- **Format**: GGUF (`.gguf`)
- **Engine**: Executed via `llama-cpp-python` on CPU (no GPU required).
- **Target Quantization**: INT4 (e.g., `Q4_K_M` or similar 4-bit quantizations).

## Configuring a Different Model Name or Path
If your model file has a different name (e.g., `qwen2.5-3b-instruct-q4_k_m.gguf`), open `config.py` and modify the `MODEL_PATH` variable:

```python
# In config.py:
MODEL_PATH = str(MODELS_DIR / "your-model-filename.gguf")
```
