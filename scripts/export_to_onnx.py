#!/usr/bin/env python3
"""Export the mohammed/fastconformer-quran-ar NeMo ASR model to ONNX.

The model is a NeMo hybrid RNNT/CTC FastConformer model. NeMo's native
``export`` method is the most reliable first step because it preserves the
model-specific export logic and may emit multiple ONNX files for transducer
models, such as an encoder file and a decoder/joint file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export mohammed/fastconformer-quran-ar to ONNX with NVIDIA NeMo.",
    )
    parser.add_argument(
        "--model-id",
        default="mohammed/fastconformer-quran-ar",
        help="Hugging Face model ID or NeMo pretrained model name.",
    )
    parser.add_argument(
        "--output-dir",
        default="onnx",
        type=Path,
        help="Directory where ONNX files and metadata will be written.",
    )
    parser.add_argument(
        "--output-name",
        default="fastconformer-quran-ar.onnx",
        help="Base ONNX file name passed to NeMo export().",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use while exporting. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--check-onnx",
        action="store_true",
        help="Run onnx.checker on every exported ONNX file.",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip best-effort tokenizer artifact export.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return requested


def load_model(model_id: str) -> Any:
    # Import lazily so --help and syntax checks do not require NeMo to be installed.
    import nemo.collections.asr as nemo_asr

    if model_id.endswith(".nemo") or Path(model_id).exists():
        return nemo_asr.models.ASRModel.restore_from(restore_path=model_id)
    return nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)


def exported_onnx_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("*.onnx"))


def write_metadata(output_dir: Path, model_id: str, device: str, files: list[Path]) -> None:
    metadata = {
        "model_id": model_id,
        "device": device,
        "onnx_files": [path.name for path in files],
        "note": (
            "Hybrid RNNT/CTC NeMo models can export more than one ONNX file; "
            "keep all generated files together for inference."
        ),
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_tokenizer_artifacts(model: Any, output_dir: Path) -> None:
    """Best-effort copy/save of tokenizer files needed by downstream runtimes."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(tokenizer, "save_to"):
        tokenizer.save_to(str(tokenizer_dir))
        return

    tokenizer_model = getattr(tokenizer, "tokenizer_model", None) or getattr(
        tokenizer, "model_path", None
    )
    if tokenizer_model and Path(tokenizer_model).exists():
        target = tokenizer_dir / Path(tokenizer_model).name
        target.write_bytes(Path(tokenizer_model).read_bytes())

    vocab = getattr(tokenizer, "vocab", None)
    if vocab is not None:
        (tokenizer_dir / "vocab.json").write_text(
            json.dumps(vocab, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def check_onnx_files(files: list[Path]) -> None:
    import onnx

    for file_path in files:
        print(f"Checking {file_path} ...")
        onnx.checker.check_model(str(file_path))


def main() -> None:
    args = parse_args()

    import torch

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name

    print(f"Loading {args.model_id} ...")
    model = load_model(args.model_id)
    model.eval()
    model.to(device)

    print(f"Exporting ONNX to {output_path} on {device} ...")
    with torch.no_grad():
        model.export(str(output_path))

    files = exported_onnx_files(args.output_dir)
    if not files:
        raise RuntimeError(
            f"NeMo export completed but no .onnx files were found in {args.output_dir}."
        )

    if not args.no_tokenizer:
        save_tokenizer_artifacts(model, args.output_dir)

    if args.check_onnx:
        check_onnx_files(files)

    write_metadata(args.output_dir, args.model_id, device, files)
    print("Export complete:")
    for file_path in files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
