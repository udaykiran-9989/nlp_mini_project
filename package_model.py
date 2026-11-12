"""Package DistilGPT2 weights, tokenizer, and custom handler into a .mar archive."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilgpt2"
MODEL_ARCHIVE_NAME = "story_continuation"
MODEL_VERSION = "1.0"


def package_model(
    project_root: Path | None = None,
    staging_dir: Path | None = None,
    export_dir: Path | None = None,
) -> Path:
    """Download/save DistilGPT2 locally and create story_continuation.mar."""
    root = project_root or Path(__file__).resolve().parent.parent
    stage = staging_dir or (root / "distilgpt2_model")
    export = export_dir or (root / "model_store")
    handler_path = root / "torchserve" / "handler.py"

    stage.mkdir(parents=True, exist_ok=True)
    export.mkdir(parents=True, exist_ok=True)

    print(f"Saving '{MODEL_ID}' tokenizer and model to {stage}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(stage)
    model.save_pretrained(stage)

    # Identify weights file as --serialized-file and remaining files as --extra-files
    weight_candidates = ["model.safetensors", "pytorch_model.bin"]
    serialized_file: Path | None = None
    for candidate in weight_candidates:
        candidate_path = stage / candidate
        if candidate_path.is_file():
            serialized_file = candidate_path
            break

    if serialized_file is None:
        raise FileNotFoundError(f"No model weights file found in {stage}.")

    extra_files = [
        str(p)
        for p in sorted(stage.iterdir())
        if p.is_file() and p.name != serialized_file.name
    ]

    cmd = [
        "torch-model-archiver",
        "--model-name",
        MODEL_ARCHIVE_NAME,
        "--version",
        MODEL_VERSION,
        "--serialized-file",
        str(serialized_file),
        "--handler",
        str(handler_path),
        "--extra-files",
        ",".join(extra_files),
        "--export-path",
        str(export),
        "--force",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    mar_path = export / f"{MODEL_ARCHIVE_NAME}.mar"
    if not mar_path.is_file():
        raise FileNotFoundError(f"Expected model archive not created at {mar_path}")

    # Clean up staging directory after packaging into .mar
    shutil.rmtree(stage, ignore_errors=True)
    print(f"Created model archive: {mar_path} ({mar_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return mar_path


if __name__ == "__main__":
    package_model()
