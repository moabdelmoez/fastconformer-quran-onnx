# FastConformer Quran Arabic ONNX export

This repository contains a small, reproducible export wrapper for converting
[`mohammed/fastconformer-quran-ar`](https://huggingface.co/mohammed/fastconformer-quran-ar)
from NVIDIA NeMo format to ONNX.

The source model is a hybrid RNNT/CTC FastConformer ASR model. NeMo's native
`model.export(...)` path is used because it owns the architecture-specific ONNX
export behavior. For transducer-style models, NeMo may produce more than one
ONNX file, commonly separate encoder and decoder/joint files. Keep all generated
ONNX files together with the tokenizer artifacts.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you have a CUDA-enabled PyTorch environment, install the PyTorch build that
matches your CUDA runtime before installing the rest of the requirements.

## Export

```bash
python scripts/export_to_onnx.py --output-dir onnx --check-onnx
```

Useful options:

```bash
python scripts/export_to_onnx.py --device cpu
python scripts/export_to_onnx.py --device cuda
python scripts/export_to_onnx.py --model-id /path/to/model.nemo --output-dir onnx
```

The script writes:

- one or more `*.onnx` files produced by NeMo;
- `export_metadata.json` describing the export;
- best-effort tokenizer artifacts under `tokenizer/`.


## CPU vs GPU for export

A GPU is helpful, but not strictly required. The ONNX exporter traces the model
with synthetic inputs, so CUDA can make the restore/export step faster for this
large FastConformer model. Use GPU when you already have a working CUDA-enabled
PyTorch and NeMo environment.

CPU export is usually simpler and more portable, but it can take much longer and
needs enough system memory for the restored model and export graph. If CUDA setup
is not already available, start with CPU to avoid spending time debugging driver
or wheel mismatches.

Recommended commands:

```bash
# Prefer this on a machine with a verified CUDA PyTorch install.
python scripts/export_to_onnx.py --device cuda --output-dir onnx --check-onnx

# Use this for maximum setup compatibility.
python scripts/export_to_onnx.py --device cpu --output-dir onnx --check-onnx
```

## Why export can be slow

The model has to be downloaded, restored by NeMo, moved to the selected device,
and traced through NeMo's ONNX exporter. This is expected to take longer on CPU,
with slow network access, or when the exporter emits multiple transducer
components.

## Notes

- Start with the default export before attempting graph surgery or quantization.
- Run with `--check-onnx` to validate exported files with `onnx.checker`.
- For deployment, use the tokenizer and all ONNX files generated from the same
  export run.
