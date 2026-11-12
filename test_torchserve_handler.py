"""Unit tests for torchserve/handler.py (Phase 3)."""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from torchserve.handler import (
    LENGTH_TOKEN_LIMITS,
    MAX_CONTEXT_TOKENS,
    PredictionException,
    StoryContinuationHandler,
    handle,
)


def _mock_hf_objects(
    prompt_len: int = 5,
    generated_len: int = 6,
    decoded: str = "revealed a brass key inside.",
):
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 50256
    tokenizer.pad_token_id = 50256

    def _tok(text, return_tensors="pt", truncation=True, max_length=1024, padding=False):
        return {
            "input_ids": torch.tensor([[10 + i for i in range(prompt_len)]], dtype=torch.long),
            "attention_mask": torch.ones((1, prompt_len), dtype=torch.long),
        }

    tokenizer.side_effect = _tok
    tokenizer.decode.return_value = f" {decoded} "

    model = MagicMock()
    model.config = types.SimpleNamespace(pad_token_id=None)
    model.to.return_value = model
    model.eval.return_value = model
    full_ids = list(range(10, 10 + prompt_len)) + list(range(300, 300 + generated_len))
    model.generate.return_value = torch.tensor([full_ids], dtype=torch.long)
    return tokenizer, model


class TestTorchServeHandler(unittest.TestCase):
    """Offline tests for StoryContinuationHandler and module-level handle()."""

    @patch("torchserve.handler.AutoModelForCausalLM.from_pretrained")
    @patch("torchserve.handler.AutoTokenizer.from_pretrained")
    def test_initialize_prefers_packaged_model_dir(
        self,
        mock_tok: MagicMock,
        mock_mod: MagicMock,
    ) -> None:
        tokenizer, model = _mock_hf_objects()
        mock_tok.return_value = tokenizer
        mock_mod.return_value = model

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").write_text("{}", encoding="utf-8")
            context = types.SimpleNamespace(system_properties={"model_dir": tmpdir})

            handler = StoryContinuationHandler()
            handler.initialize(context)

            self.assertTrue(handler.initialized)
            mock_tok.assert_called_once_with(tmpdir)
            mock_mod.assert_called_once_with(tmpdir)
            self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
            model.eval.assert_called_once()

    @patch("torchserve.handler.AutoModelForCausalLM.from_pretrained")
    @patch("torchserve.handler.AutoTokenizer.from_pretrained")
    def test_valid_request_lengths_and_temperatures(
        self,
        mock_tok: MagicMock,
        mock_mod: MagicMock,
    ) -> None:
        tokenizer, model = _mock_hf_objects(prompt_len=4, generated_len=8)
        mock_tok.return_value = tokenizer
        mock_mod.return_value = model

        handler = StoryContinuationHandler()
        handler.initialize(None)

        for length_name, expected_tokens in LENGTH_TOKEN_LIMITS.items():
            for temp in (0.2, 0.8, 1.4):
                payload = json.dumps(
                    {
                        "prompt": "One day, a student discovered a mysterious box...",
                        "length": length_name,
                        "temperature": temp,
                    }
                ).encode("utf-8")
                processed = handler.preprocess([{"body": payload}])
                raw_out = handler.inference(processed)
                res = handler.postprocess(raw_out)

                self.assertEqual(len(res), 1)
                self.assertEqual(res[0]["generated_text"], "revealed a brass key inside.")
                self.assertEqual(res[0]["length"], length_name)
                self.assertEqual(res[0]["temperature"], temp)

                gen_kwargs = model.generate.call_args.kwargs
                self.assertEqual(gen_kwargs["max_new_tokens"], expected_tokens)
                self.assertEqual(gen_kwargs["temperature"], temp)
                self.assertTrue(gen_kwargs["do_sample"])

        # Only the 8 newly generated tokens should be decoded
        decoded_ids = tokenizer.decode.call_args.args[0].tolist()
        self.assertEqual(decoded_ids, list(range(300, 308)))

    @patch("torchserve.handler.AutoModelForCausalLM.from_pretrained")
    @patch("torchserve.handler.AutoTokenizer.from_pretrained")
    def test_very_long_prompt_sets_truncation_limit(
        self,
        mock_tok: MagicMock,
        mock_mod: MagicMock,
    ) -> None:
        tokenizer, model = _mock_hf_objects(prompt_len=844, generated_len=5)
        mock_tok.return_value = tokenizer
        mock_mod.return_value = model

        handler = StoryContinuationHandler()
        handler.initialize(None)

        long_prompt = "story " * 1500
        processed = handler.preprocess(
            [{"body": {"prompt": long_prompt, "length": "Long", "temperature": 0.8}}]
        )
        self.assertEqual(len(processed), 1)
        tok_kwargs = tokenizer.call_args.kwargs
        self.assertTrue(tok_kwargs["truncation"])
        self.assertEqual(
            tok_kwargs["max_length"],
            MAX_CONTEXT_TOKENS - LENGTH_TOKEN_LIMITS["Long"],
        )

    @patch("torchserve.handler.AutoModelForCausalLM.from_pretrained")
    @patch("torchserve.handler.AutoTokenizer.from_pretrained")
    def test_validation_errors_for_bad_inputs(
        self,
        mock_tok: MagicMock,
        mock_mod: MagicMock,
    ) -> None:
        tokenizer, model = _mock_hf_objects()
        mock_tok.return_value = tokenizer
        mock_mod.return_value = model

        handler = StoryContinuationHandler()
        handler.initialize(None)

        bad_bodies = [
            b"",
            b"{not-json",
            json.dumps({}),
            json.dumps({"length": "Medium", "temperature": 0.8}),
            json.dumps({"prompt": "   "}),
            json.dumps({"prompt": "Hello", "length": "Huge"}),
            json.dumps({"prompt": "Hello", "temperature": 0.05}),
            json.dumps({"prompt": "Hello", "temperature": 1.6}),
            json.dumps({"prompt": "Hello", "temperature": "warm"}),
            json.dumps({"prompt": "Hello", "temperature": True}),
        ]
        for body in bad_bodies:
            with self.assertRaises(PredictionException) as ctx:
                handler.preprocess([{"body": body}])
            self.assertEqual(ctx.exception.error_code, 400)

    @patch("torchserve.handler.AutoModelForCausalLM.from_pretrained")
    @patch("torchserve.handler.AutoTokenizer.from_pretrained")
    def test_module_handle_entrypoint(
        self,
        mock_tok: MagicMock,
        mock_mod: MagicMock,
    ) -> None:
        tokenizer, model = _mock_hf_objects()
        mock_tok.return_value = tokenizer
        mock_mod.return_value = model

        import torchserve.handler as handler_mod

        handler_mod._service = StoryContinuationHandler()
        res = handle(
            [{"body": json.dumps({"prompt": "A quiet library stood on the hill."})}],
            None,
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["length"], "Medium")
        self.assertEqual(res[0]["temperature"], 0.8)
        self.assertEqual(res[0]["generated_text"], "revealed a brass key inside.")


if __name__ == "__main__":
    unittest.main()
