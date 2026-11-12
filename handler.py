"""Custom TorchServe handler for DistilGPT2 story continuation (Phase 3)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from ts.utils.util import PredictionException
except ImportError:
    class PredictionException(Exception):
        """Fallback exception when running unit tests outside TorchServe."""

        def __init__(self, message: str, error_code: int = 500) -> None:
            super().__init__(message)
            self.message = message
            self.error_code = error_code


DEFAULT_MODEL_NAME = "distilgpt2"
MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 1.5
DEFAULT_TEMPERATURE = 0.8
DEFAULT_LENGTH = "Medium"

LENGTH_TOKEN_LIMITS: dict[str, int] = {
    "Short": 50,
    "Medium": 100,
    "Long": 180,
}

MAX_CONTEXT_TOKENS = 1024
DEFAULT_TOP_P = 0.92
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3


class StoryContinuationHandler:
    """TorchServe inference handler for pretrained DistilGPT2."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: torch.device = torch.device("cpu")
        self.initialized: bool = False

    @staticmethod
    def _resolve_model_source(context: Any) -> str:
        """Locate packaged model files in `model_dir` or fall back to HF Hub ID."""
        if context is not None:
            properties = getattr(context, "system_properties", {}) or {}
            model_dir = properties.get("model_dir")
            if model_dir:
                base_path = Path(model_dir)
                if (base_path / "config.json").is_file():
                    return str(base_path)
                nested_dir = base_path / "distilgpt2_model"
                if (nested_dir / "config.json").is_file():
                    return str(nested_dir)
        return DEFAULT_MODEL_NAME

    def initialize(self, context: Any) -> None:
        """Load DistilGPT2 tokenizer and causal LM once at worker startup."""
        self.device = torch.device("cpu")
        model_source = self._resolve_model_source(context)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = AutoModelForCausalLM.from_pretrained(model_source)

            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if (
                getattr(self.model.config, "pad_token_id", None) is None
                and self.tokenizer.pad_token_id is not None
            ):
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
        except Exception as exc:
            self.initialized = False
            raise PredictionException(
                f"Failed to initialize DistilGPT2 model from '{model_source}': {exc}",
                500,
            ) from exc

    @staticmethod
    def _parse_json_body(row: dict[str, Any]) -> dict[str, Any]:
        """Extract and parse JSON payload from a TorchServe request item."""
        if not isinstance(row, dict):
            raise PredictionException("Invalid request structure.", 400)

        raw = row.get("body")
        if raw is None:
            raw = row.get("data")

        if raw is None:
            raise PredictionException("Request body is empty; JSON object required.", 400)

        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PredictionException("Request body must be valid UTF-8.", 400) from exc

        if isinstance(raw, str):
            raw_stripped = raw.strip()
            if not raw_stripped:
                raise PredictionException("Request body is empty; JSON object required.", 400)
            try:
                parsed = json.loads(raw_stripped)
            except json.JSONDecodeError as exc:
                raise PredictionException(f"Malformed JSON payload: {exc.msg}", 400) from exc
        elif isinstance(raw, dict):
            parsed = raw
        else:
            raise PredictionException("Unsupported request payload type; expected JSON object.", 400)

        if not isinstance(parsed, dict):
            raise PredictionException("JSON payload must be an object.", 400)

        return parsed

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate prompt, length, and temperature fields."""
        if "prompt" not in payload:
            raise PredictionException("Missing required field: 'prompt'.", 400)

        prompt = payload["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise PredictionException("Field 'prompt' must be a non-empty string.", 400)

        length_option = payload.get("length", DEFAULT_LENGTH)
        if not isinstance(length_option, str) or length_option not in LENGTH_TOKEN_LIMITS:
            valid = ", ".join(LENGTH_TOKEN_LIMITS.keys())
            raise PredictionException(
                f"Invalid 'length' value '{length_option}'. Choose from: {valid}.",
                400,
            )

        temp_raw = payload.get("temperature", DEFAULT_TEMPERATURE)
        if isinstance(temp_raw, bool):
            raise PredictionException("Field 'temperature' must be numeric.", 400)
        try:
            temperature = float(temp_raw)
        except (TypeError, ValueError) as exc:
            raise PredictionException("Field 'temperature' must be numeric.", 400) from exc

        if not math.isfinite(temperature) or not (MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE):
            raise PredictionException(
                f"Field 'temperature' must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}.",
                400,
            )

        return {
            "prompt": prompt.strip(),
            "length": length_option,
            "max_new_tokens": LENGTH_TOKEN_LIMITS[length_option],
            "temperature": round(temperature, 2),
        }

    def preprocess(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Parse, validate, and tokenize incoming TorchServe requests."""
        if not data:
            raise PredictionException("Empty request batch.", 400)

        if not self.initialized or self.tokenizer is None:
            raise PredictionException("Handler tokenizer is not initialized.", 500)

        processed: list[dict[str, Any]] = []
        for row in data:
            payload = self._parse_json_body(row)
            validated = self._validate_payload(payload)

            max_input_tokens = max(1, MAX_CONTEXT_TOKENS - validated["max_new_tokens"])
            encoded = self.tokenizer(
                validated["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
                padding=False,
            )
            if "attention_mask" not in encoded or encoded["attention_mask"] is None:
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])

            processed.append(
                {
                    "input_ids": encoded["input_ids"].to(self.device),
                    "attention_mask": encoded["attention_mask"].to(self.device),
                    "length": validated["length"],
                    "max_new_tokens": validated["max_new_tokens"],
                    "temperature": validated["temperature"],
                }
            )

        return processed

    def inference(self, model_input: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate story continuations using DistilGPT2 in inference mode."""
        if not self.initialized or self.model is None or self.tokenizer is None:
            raise PredictionException("Handler model is not initialized.", 500)

        outputs: list[dict[str, Any]] = []
        try:
            for item in model_input:
                input_ids = item["input_ids"]
                attention_mask = item["attention_mask"]
                prompt_len = int(input_ids.shape[1])
                max_new_tokens = item["max_new_tokens"]

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        min_new_tokens=min(15, max_new_tokens),
                        max_new_tokens=max_new_tokens,
                        temperature=item["temperature"],
                        do_sample=True,
                        top_p=DEFAULT_TOP_P,
                        repetition_penalty=DEFAULT_REPETITION_PENALTY,
                        no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                newly_generated_ids = output_ids[0][prompt_len:]
                continuation = self.tokenizer.decode(
                    newly_generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()

                outputs.append(
                    {
                        "generated_text": continuation,
                        "length": item["length"],
                        "temperature": item["temperature"],
                    }
                )
        except PredictionException:
            raise
        except Exception as exc:
            raise PredictionException(f"Story generation inference failed: {exc}", 500) from exc

        return outputs

    def postprocess(self, inference_output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return structured JSON response dictionaries for TorchServe."""
        return inference_output


_service = StoryContinuationHandler()


def handle(data: list[dict[str, Any]] | None, context: Any) -> list[dict[str, Any]]:
    """TorchServe entrypoint function."""
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return []

    model_input = _service.preprocess(data)
    inference_output = _service.inference(model_input)
    return _service.postprocess(inference_output)
