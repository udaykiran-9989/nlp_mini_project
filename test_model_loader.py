"""Unit tests for model/model_loader.py (Phase 2: DistilGPT2 with mocked HF classes)."""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from model.model_loader import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    clear_model_cache,
    generate_story_continuation,
    join_prompt_and_continuation,
    load_model,
    resolve_generation_config,
    tokenize_prompt,
    validate_generation_params,
    validate_story_prompt,
)


def _build_mock_tokenizer_and_model(
    prompt_len: int = 4,
    generated_len: int = 6,
    decoded_text: str = "a glowing key rested beneath the velvet lining.",
):
    """Create lightweight mock tokenizer and causal LM objects for offline tests."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 50256
    tokenizer.pad_token_id = 50256

    def _tokenize(text, return_tensors="pt", truncation=True, max_length=1024, padding=False):
        return {
            "input_ids": torch.tensor([[101 + i for i in range(prompt_len)]], dtype=torch.long),
            "attention_mask": torch.ones((1, prompt_len), dtype=torch.long),
        }

    tokenizer.side_effect = _tokenize
    tokenizer.decode.return_value = f" {decoded_text} "

    model = MagicMock()
    model.config = types.SimpleNamespace(pad_token_id=None)
    model.to.return_value = model
    model.eval.return_value = model

    full_ids = list(range(101, 101 + prompt_len)) + list(range(200, 200 + generated_len))
    model.generate.return_value = torch.tensor([full_ids], dtype=torch.long)
    return tokenizer, model


class TestModelLoaderPhase2(unittest.TestCase):
    """Tests covering model loading, tokenization, generation, and validation."""

    def setUp(self) -> None:
        clear_model_cache()

    def tearDown(self) -> None:
        clear_model_cache()

    def test_validate_story_prompt_rejects_empty_and_none(self) -> None:
        for invalid in (None, "", "   ", "\n\t  "):
            is_valid, message = validate_story_prompt(invalid)
            self.assertFalse(is_valid)
            self.assertTrue(len(message) > 0)

    def test_validate_story_prompt_accepts_non_empty(self) -> None:
        is_valid, message = validate_story_prompt("Once upon a time...")
        self.assertTrue(is_valid)
        self.assertEqual(message, "")

    def test_validate_generation_params_and_invalid_settings(self) -> None:
        length, temp = validate_generation_params("Short", 0.1)
        self.assertEqual(length, "Short")
        self.assertEqual(temp, 0.1)

        with self.assertRaises(ValueError):
            validate_generation_params("ExtraLong", 0.8)

        with self.assertRaises(ValueError):
            validate_generation_params("Medium", MIN_TEMPERATURE - 0.05)

        with self.assertRaises(ValueError):
            validate_generation_params("Medium", MAX_TEMPERATURE + 0.1)

        with self.assertRaises(ValueError):
            validate_generation_params("Medium", "invalid")  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            resolve_generation_config("Short", 0.8, max_new_tokens=0)

        with self.assertRaises(ValueError):
            resolve_generation_config("Short", 0.8, max_new_tokens=True)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            resolve_generation_config("Short", 0.8, repetition_penalty=0.9)

        with self.assertRaises(ValueError):
            resolve_generation_config("Short", 0.8, no_repeat_ngram_size=-1)

    @patch("model.model_loader.AutoModelForCausalLM.from_pretrained")
    @patch("model.model_loader.AutoTokenizer.from_pretrained")
    def test_load_model_sets_padding_eval_cpu_and_caches(
        self,
        mock_tok_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
    ) -> None:
        tokenizer, model = _build_mock_tokenizer_and_model()
        mock_tok_from_pretrained.return_value = tokenizer
        mock_model_from_pretrained.return_value = model

        tok1, mod1 = load_model()
        tok2, mod2 = load_model()

        self.assertIs(tok1, tok2)
        self.assertIs(mod1, mod2)
        mock_tok_from_pretrained.assert_called_once_with(DEFAULT_MODEL_NAME)
        mock_model_from_pretrained.assert_called_once_with(DEFAULT_MODEL_NAME)
        self.assertEqual(tok1.pad_token, tok1.eos_token)
        self.assertEqual(mod1.config.pad_token_id, tok1.pad_token_id)
        mod1.to.assert_called_once_with("cpu")
        mod1.eval.assert_called_once()

    @patch("model.model_loader.AutoTokenizer.from_pretrained", side_effect=OSError("DNS failure"))
    def test_load_model_raises_runtime_error_on_failure(
        self,
        mock_tok_from_pretrained: MagicMock,
    ) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            load_model()
        self.assertIn("Failed to load model", str(ctx.exception))
        mock_tok_from_pretrained.assert_called_once()

    def test_tokenize_prompt_returns_tensors_and_mask(self) -> None:
        tokenizer, _ = _build_mock_tokenizer_and_model(prompt_len=5)
        encoded = tokenize_prompt("A quiet library.", tokenizer, max_input_tokens=512)
        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)
        self.assertEqual(tuple(encoded["input_ids"].shape), (1, 5))
        self.assertEqual(tuple(encoded["attention_mask"].shape), (1, 5))

    @patch("model.model_loader.AutoModelForCausalLM.from_pretrained")
    @patch("model.model_loader.AutoTokenizer.from_pretrained")
    def test_generate_story_continuation_lengths_and_temperatures(
        self,
        mock_tok_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
    ) -> None:
        tokenizer, model = _build_mock_tokenizer_and_model(
            prompt_len=4,
            generated_len=7,
            decoded_text="inside lay an antique brass compass.",
        )
        mock_tok_from_pretrained.return_value = tokenizer
        mock_model_from_pretrained.return_value = model

        expected_limits = {"Short": 50, "Medium": 100, "Long": 180}
        temps = [0.2, DEFAULT_TEMPERATURE, 1.4]

        for length_opt, temp in zip(expected_limits.keys(), temps):
            result = generate_story_continuation(
                "One day, a student discovered a mysterious box in the college library...",
                length_option=length_opt,
                temperature=temp,
            )
            self.assertFalse(result["is_placeholder"])
            self.assertEqual(result["model_name"], DEFAULT_MODEL_NAME)
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["length_option"], length_opt)
            self.assertEqual(result["max_new_tokens"], expected_limits[length_opt])
            self.assertEqual(result["temperature"], temp)
            self.assertEqual(result["generated_token_count"], 7)
            self.assertEqual(
                result["prompt"],
                "One day, a student discovered a mysterious box in the college library...",
            )
            self.assertEqual(result["continuation"], "inside lay an antique brass compass.")
            self.assertIn(result["continuation"], result["full_story"])

            call_kwargs = model.generate.call_args.kwargs
            self.assertTrue(call_kwargs["do_sample"])
            self.assertEqual(call_kwargs["max_new_tokens"], expected_limits[length_opt])
            self.assertEqual(call_kwargs["temperature"], temp)
            self.assertGreater(call_kwargs["repetition_penalty"], 1.0)
            self.assertGreaterEqual(call_kwargs["no_repeat_ngram_size"], 2)

        # Verify decode received only the 7 newly generated tokens (IDs 200..206)
        decoded_arg = tokenizer.decode.call_args.args[0]
        self.assertEqual(decoded_arg.tolist(), [200, 201, 202, 203, 204, 205, 206])

    def test_generate_raises_on_empty_prompt_and_invalid_settings(self) -> None:
        with self.assertRaises(ValueError):
            generate_story_continuation("   ", "Medium", 0.8)

        with self.assertRaises(ValueError):
            generate_story_continuation("Hello", "Huge", 0.8)

        with self.assertRaises(ValueError):
            generate_story_continuation("Hello", "Short", 2.0)

    def test_join_prompt_and_continuation_punctuation(self) -> None:
        self.assertEqual(
            join_prompt_and_continuation("She opened the door", ", revealing stairs."),
            "She opened the door, revealing stairs.",
        )
        self.assertEqual(
            join_prompt_and_continuation("She opened the door", "and stepped inside."),
            "She opened the door and stepped inside.",
        )


if __name__ == "__main__":
    unittest.main()
