"""Unit tests for app/streamlit_app.py state, backend routing, and error handling."""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests


def _load_streamlit_app_with_mock():
    """Import app.streamlit_app using a lightweight mock if streamlit is unavailable."""
    if "streamlit" not in sys.modules:
        fake_st = MagicMock()
        fake_components = types.ModuleType("streamlit.components")
        fake_components_v1 = types.ModuleType("streamlit.components.v1")
        fake_components_v1.html = MagicMock()
        fake_components.v1 = fake_components_v1

        sys.modules["streamlit"] = fake_st
        sys.modules["streamlit.components"] = fake_components
        sys.modules["streamlit.components.v1"] = fake_components_v1

    import app.streamlit_app as streamlit_app

    return streamlit_app


class TestStreamlitAppHelpers(unittest.TestCase):
    """Tests for session state lifecycle, backend routing, and syntax validity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ui = _load_streamlit_app_with_mock()

    def test_streamlit_app_parses_cleanly(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
        source = app_path.read_text(encoding="utf-8")
        ast.parse(source)

    def test_initialize_and_reset_session_state(self) -> None:
        state: dict = {}
        self.ui.initialize_session_state(state)

        self.assertEqual(state["story_prompt"], "")
        self.assertEqual(state["response_length"], "Medium")
        self.assertEqual(state["temperature"], 0.8)
        self.assertEqual(state["inference_backend"], self.ui.DEFAULT_BACKEND)
        self.assertEqual(state["torchserve_url"], self.ui.DEFAULT_TORCHSERVE_URL)
        self.assertIsNone(state["generated_result"])
        self.assertIsNone(state["validation_error"])
        self.assertIsNone(state["generation_error"])
        self.assertIsNone(state["backend_notice"])

        state["story_prompt"] = "Once upon a time"
        state["response_length"] = "Long"
        state["temperature"] = 1.2
        state["generated_result"] = {"full_story": "Once upon a time..."}
        state["validation_error"] = "old validation error"
        state["generation_error"] = "old runtime error"
        state["backend_notice"] = "old fallback notice"

        self.ui.reset_session_state(state)
        self.assertEqual(state["story_prompt"], "")
        self.assertEqual(state["response_length"], "Medium")
        self.assertEqual(state["temperature"], 0.8)
        self.assertIsNone(state["generated_result"])
        self.assertIsNone(state["validation_error"])
        self.assertIsNone(state["generation_error"])
        self.assertIsNone(state["backend_notice"])

    def test_handle_generate_request_local_backend_success(self) -> None:
        result, err = self.ui.handle_generate_request("   ", "Medium", 0.8)
        self.assertIsNone(result)
        self.assertIsNotNone(err)

        mock_output = {
            "prompt": "In a quiet village beside the sea,",
            "continuation": "lanterns flickered along the stone pier.",
            "full_story": "In a quiet village beside the sea, lanterns flickered along the stone pier.",
            "length_option": "Short",
            "temperature": 0.8,
            "is_placeholder": False,
        }
        with patch.object(self.ui, "generate_story_continuation", return_value=mock_output) as mock_gen:
            result, err = self.ui.handle_generate_request(
                "In a quiet village beside the sea,",
                "Short",
                0.8,
                backend=self.ui.BACKEND_LOCAL,
            )
            mock_gen.assert_called_once_with(
                prompt="In a quiet village beside the sea,",
                length_option="Short",
                temperature=0.8,
            )

        self.assertIsNone(err)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["backend"], "Local DistilGPT2")

    def test_handle_generate_request_torchserve_api_success(self) -> None:
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "generated_text": "she found an ancient map folded inside.",
            "length": "Medium",
            "temperature": 0.8,
        }

        with patch.object(self.ui.requests, "post", return_value=fake_response) as mock_post:
            result, err = self.ui.handle_generate_request(
                "One day, a student discovered a mysterious box...",
                "Medium",
                0.8,
                backend=self.ui.BACKEND_TORCHSERVE_ONLY,
                torchserve_url="http://localhost:8090/predictions/story_continuation",
            )

        mock_post.assert_called_once()
        self.assertIsNone(err)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["backend"], "TorchServe API")
        self.assertEqual(result["continuation"], "she found an ancient map folded inside.")

    def test_handle_generate_request_torchserve_errors_and_fallback(self) -> None:
        # Strict TorchServe mode returns clean error when API is unreachable
        with patch.object(
            self.ui.requests,
            "post",
            side_effect=requests.ConnectionError("Connection refused"),
        ):
            result, err = self.ui.handle_generate_request(
                "A cold wind blew.",
                "Short",
                0.8,
                backend=self.ui.BACKEND_TORCHSERVE_ONLY,
            )
        self.assertIsNone(result)
        self.assertIn("Could not reach TorchServe", str(err))

        # Auto-fallback mode falls back to local DistilGPT2 when TorchServe is unreachable
        fallback_output = {
            "prompt": "A cold wind blew.",
            "continuation": "across the empty courtyard.",
            "full_story": "A cold wind blew. across the empty courtyard.",
            "length_option": "Short",
            "temperature": 0.8,
            "is_placeholder": False,
        }
        with (
            patch.object(
                self.ui.requests,
                "post",
                side_effect=requests.ConnectionError("Connection refused"),
            ),
            patch.object(self.ui, "generate_story_continuation", return_value=fallback_output),
        ):
            result, err = self.ui.handle_generate_request(
                "A cold wind blew.",
                "Short",
                0.8,
                backend=self.ui.BACKEND_TORCHSERVE_AUTO,
            )
        self.assertIsNone(err)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["backend"], "Local Fallback")
        self.assertIn("fallback_reason", result)

    def test_handle_generate_request_surfaces_runtime_error(self) -> None:
        with patch.object(
            self.ui,
            "generate_story_continuation",
            side_effect=RuntimeError("Model cache unavailable"),
        ):
            result, err = self.ui.handle_generate_request(
                "A cold wind blew.",
                "Short",
                0.8,
                backend=self.ui.BACKEND_LOCAL,
            )

        self.assertIsNone(result)
        self.assertEqual(err, "Model cache unavailable")


if __name__ == "__main__":
    unittest.main()
