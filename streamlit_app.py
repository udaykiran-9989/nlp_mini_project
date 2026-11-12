"""Streamlit web interface for the AI Story Continuation System (Phase 3: TorchServe + Local).

Run from the project root using:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any, MutableMapping

import requests
import streamlit as st
import streamlit.components.v1 as components

# Ensure the project root is importable when executed via `streamlit run app/streamlit_app.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model_loader import (  # noqa: E402
    DEFAULT_TEMPERATURE,
    LENGTH_PRESETS,
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    generate_story_continuation,
    join_prompt_and_continuation,
    validate_generation_params,
    validate_story_prompt,
)

APP_TITLE = "AI Story Continuation System"
DEFAULT_PROMPT = ""
DEFAULT_LENGTH = "Medium"
LENGTH_OPTIONS = list(LENGTH_PRESETS.keys())

BACKEND_TORCHSERVE_AUTO = "TorchServe API (Auto Fallback)"
BACKEND_TORCHSERVE_ONLY = "TorchServe API"
BACKEND_LOCAL = "Local DistilGPT2"
BACKEND_OPTIONS = [BACKEND_TORCHSERVE_AUTO, BACKEND_TORCHSERVE_ONLY, BACKEND_LOCAL]

DEFAULT_TORCHSERVE_URL = os.getenv(
    "TORCHSERVE_URL",
    "http://localhost:8090/predictions/story_continuation",
)
DEFAULT_TORCHSERVE_TIMEOUT = int(os.getenv("TORCHSERVE_TIMEOUT", "60"))


def _resolve_default_backend() -> str:
    """Resolve initial inference backend from INFERENCE_BACKEND env var."""
    raw = os.getenv("INFERENCE_BACKEND", "").strip().lower()
    mapping = {
        "torchserve": BACKEND_TORCHSERVE_AUTO,
        "torchserve_only": BACKEND_TORCHSERVE_ONLY,
        "api": BACKEND_TORCHSERVE_ONLY,
        "local": BACKEND_LOCAL,
    }
    if raw in mapping:
        return mapping[raw]
    for opt in BACKEND_OPTIONS:
        if raw == opt.lower():
            return opt
    return BACKEND_TORCHSERVE_AUTO


DEFAULT_BACKEND = _resolve_default_backend()

BOOK_ILLUSTRATION_SVG = """
<svg width="56" height="56" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect x="4" y="4" width="56" height="56" rx="16" fill="#F3EEFF"/>
  <path d="M16 21.5C16 19.567 17.567 18 19.5 18H29.5C30.8807 18 32 19.1193 32 20.5V45.5C32 44.1193 30.8807 43 29.5 43H19.5C17.567 43 16 44.567 16 46.5V21.5Z" fill="#FFFFFF" stroke="#7C3AED" stroke-width="2.2" stroke-linejoin="round"/>
  <path d="M48 21.5C48 19.567 46.433 18 44.5 18H34.5C33.1193 18 32 19.1193 32 20.5V45.5C32 44.1193 33.1193 43 34.5 43H44.5C46.433 43 48 44.567 48 46.5V21.5Z" fill="#FAF7FF" stroke="#7C3AED" stroke-width="2.2" stroke-linejoin="round"/>
  <path d="M20.5 24.5H27.5M20.5 29.5H27.5M20.5 34.5H26" stroke="#A78BFA" stroke-width="2" stroke-linecap="round"/>
  <path d="M36.5 24.5H43.5M36.5 29.5H43.5M36.5 34.5H42" stroke="#A78BFA" stroke-width="2" stroke-linecap="round"/>
  <path d="M47 13L48.1 15.6L50.8 16.7L48.1 17.8L47 20.4L45.9 17.8L43.2 16.7L45.9 15.6L47 13Z" fill="#7C3AED"/>
</svg>
"""


def initialize_session_state(state: MutableMapping[str, Any]) -> None:
    """Initialize default keys in Streamlit session state if missing."""
    defaults: dict[str, Any] = {
        "story_prompt": DEFAULT_PROMPT,
        "response_length": DEFAULT_LENGTH,
        "temperature": DEFAULT_TEMPERATURE,
        "inference_backend": DEFAULT_BACKEND,
        "torchserve_url": DEFAULT_TORCHSERVE_URL,
        "generated_result": None,
        "validation_error": None,
        "generation_error": None,
        "backend_notice": None,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


def reset_session_state(state: MutableMapping[str, Any]) -> None:
    """Reset prompt, controls, validation messages, and generated story output."""
    state["story_prompt"] = DEFAULT_PROMPT
    state["response_length"] = DEFAULT_LENGTH
    state["temperature"] = DEFAULT_TEMPERATURE
    state["generated_result"] = None
    state["validation_error"] = None
    state["generation_error"] = None
    state["backend_notice"] = None


def call_torchserve_api(
    prompt: str,
    length_option: str,
    temperature: float,
    endpoint_url: str = DEFAULT_TORCHSERVE_URL,
    timeout: int = DEFAULT_TORCHSERVE_TIMEOUT,
) -> dict[str, Any]:
    """Send a story continuation request to the TorchServe REST API."""
    is_valid, error_message = validate_story_prompt(prompt)
    if not is_valid:
        raise ValueError(error_message)

    normalized_length, normalized_temp = validate_generation_params(length_option, temperature)
    cleaned_prompt = prompt.strip()
    url = endpoint_url.strip()
    if not url:
        raise ValueError("TorchServe API URL cannot be empty.")

    payload = {
        "prompt": cleaned_prompt,
        "length": normalized_length,
        "temperature": normalized_temp,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Could not reach TorchServe at '{url}': {exc}"
        ) from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        try:
            err_json = response.json()
            if isinstance(err_json, dict) and err_json.get("message"):
                detail = str(err_json["message"])
        except ValueError:
            pass
        raise RuntimeError(f"TorchServe API error ({response.status_code}): {detail}")

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("TorchServe returned a non-JSON response.") from exc

    if not isinstance(data, dict) or "generated_text" not in data:
        raise RuntimeError("TorchServe response is missing 'generated_text'.")

    continuation = str(data["generated_text"]).strip()
    full_story = join_prompt_and_continuation(cleaned_prompt, continuation)

    return {
        "prompt": cleaned_prompt,
        "continuation": continuation,
        "full_story": full_story,
        "length_option": str(data.get("length", normalized_length)),
        "temperature": float(data.get("temperature", normalized_temp)),
        "model_name": "distilgpt2",
        "backend": "TorchServe API",
        "is_placeholder": False,
    }


def handle_generate_request(
    prompt: str,
    length_option: str,
    temperature: float,
    backend: str = DEFAULT_BACKEND,
    torchserve_url: str = DEFAULT_TORCHSERVE_URL,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate user input and generate a story via TorchServe API or local DistilGPT2.

    Returns:
        ``(result_dict, error_message)`` where one element is ``None``.
    """
    is_valid, error_message = validate_story_prompt(prompt)
    if not is_valid:
        return None, error_message

    if backend in (BACKEND_TORCHSERVE_AUTO, BACKEND_TORCHSERVE_ONLY):
        try:
            result = call_torchserve_api(
                prompt=prompt,
                length_option=length_option,
                temperature=temperature,
                endpoint_url=torchserve_url,
            )
            return result, None
        except ConnectionError as exc:
            if backend == BACKEND_TORCHSERVE_ONLY:
                return None, str(exc)
            # Fall back cleanly to local DistilGPT2 when TorchServe is unreachable
            try:
                local_result = generate_story_continuation(
                    prompt=prompt,
                    length_option=length_option,
                    temperature=temperature,
                )
                local_result["backend"] = "Local Fallback"
                local_result["fallback_reason"] = str(exc)
                return local_result, None
            except (ValueError, RuntimeError) as local_exc:
                return None, str(local_exc)
        except (ValueError, RuntimeError) as exc:
            return None, str(exc)

    try:
        result = generate_story_continuation(
            prompt=prompt,
            length_option=length_option,
            temperature=temperature,
        )
        result["backend"] = "Local DistilGPT2"
    except (ValueError, RuntimeError) as exc:
        return None, str(exc)

    return result, None


def _on_reset_click() -> None:
    """Streamlit button callback to reset all form and output state cleanly."""
    reset_session_state(st.session_state)


def _inject_custom_styles() -> None:
    """Apply light, lavender-accented styling matching a clean minimal UI."""
    st.markdown(
        """
        <style>
        /* Clean white canvas & typography */
        .stApp {
            background-color: #FFFFFF;
            color: #1F1B2E;
            font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .block-container {
            max-width: 760px;
            padding-top: 2.25rem;
            padding-bottom: 2.5rem;
        }

        /* Header card */
        .app-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1.25rem 1.45rem;
            margin-bottom: 1.25rem;
            background: linear-gradient(135deg, #FFFFFF 0%, #F9F7FF 100%);
            border: 1px solid #EAE4FA;
            border-radius: 16px;
            box-shadow: 0 4px 18px rgba(109, 40, 217, 0.06);
        }

        .app-header-icon {
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .app-header-title {
            margin: 0;
            font-size: 1.55rem;
            font-weight: 700;
            color: #1F1B2E;
            letter-spacing: -0.02em;
            line-height: 1.25;
        }

        .app-header-subtitle {
            margin: 0.25rem 0 0 0;
            font-size: 0.92rem;
            color: #645E7A;
            line-height: 1.45;
        }

        /* Bordered Streamlit cards (Input & Output cards) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF;
            border: 1px solid #EAE4FA !important;
            border-radius: 16px !important;
            box-shadow: 0 6px 24px rgba(109, 40, 217, 0.06);
            padding: 0.65rem 0.75rem;
        }

        /* Labels */
        .stTextArea label,
        .stSelectbox label,
        .stSlider label {
            color: #2D2645 !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
        }

        /* Text area styling */
        .stTextArea textarea {
            background-color: #FCFBFF !important;
            border: 1px solid #DDD6FE !important;
            border-radius: 12px !important;
            color: #1F1B2E !important;
            font-size: 0.96rem !important;
            line-height: 1.6 !important;
            padding: 0.85rem 1rem !important;
        }

        .stTextArea textarea:focus {
            border-color: #7C3AED !important;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.14) !important;
        }

        /* Selectbox styling */
        div[data-baseweb="select"] > div {
            background-color: #FCFBFF !important;
            border: 1px solid #DDD6FE !important;
            border-radius: 10px !important;
            color: #1F1B2E !important;
        }

        /* Primary Generate button */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.25rem !important;
            box-shadow: 0 4px 14px rgba(109, 40, 217, 0.22) !important;
        }

        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #6D28D9 0%, #5B21B6 100%) !important;
        }

        /* Secondary Reset button */
        div.stButton > button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #5B21B6 !important;
            border: 1px solid #DDD6FE !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.1rem !important;
        }

        div.stButton > button[kind="secondary"]:hover {
            background-color: #F5F3FF !important;
            border-color: #C4B5FD !important;
        }

        /* Generated Story card content */
        .story-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.85rem;
        }

        .story-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #1F1B2E;
            margin: 0;
        }

        .story-meta-group {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            flex-wrap: wrap;
        }

        .story-pill {
            display: inline-block;
            background-color: #F3EEFF;
            color: #5B21B6;
            border: 1px solid #E4DBFC;
            border-radius: 999px;
            padding: 0.18rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .story-body {
            background-color: #FAF8FF;
            border: 1px solid #EDE9FE;
            border-radius: 12px;
            padding: 1.15rem 1.25rem;
            color: #26213B;
            font-size: 0.98rem;
            line-height: 1.72;
            white-space: pre-wrap;
        }

        .story-prompt-lead {
            color: #4C1D95;
            font-weight: 600;
        }

        .story-empty-state {
            background-color: #FAF8FF;
            border: 1px dashed #DDD6FE;
            border-radius: 12px;
            padding: 1.35rem 1.2rem;
            text-align: center;
            color: #6B6484;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        /* Disclaimer footer */
        .ai-disclaimer {
            margin-top: 1.25rem;
            padding: 0.8rem 1rem;
            background-color: #FAF8FF;
            border: 1px solid #EDE9FE;
            border-radius: 12px;
            color: #6B6484;
            font-size: 0.82rem;
            line-height: 1.5;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    """Render the application header with the book illustration and title."""
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-icon">{BOOK_ILLUSTRATION_SVG}</div>
            <div>
                <h1 class="app-header-title">{APP_TITLE}</h1>
                <p class="app-header-subtitle">
                    Enter your story opening, adjust length and creativity, and preview a continued narrative.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_copy_button(text_to_copy: str) -> None:
    """Render a styled client-side Copy button using Streamlit's HTML component."""
    safe_json_text = json.dumps(text_to_copy)
    button_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body {{
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .copy-btn {{
          width: 100%;
          height: 39px;
          background-color: #FFFFFF;
          color: #5B21B6;
          border: 1px solid #DDD6FE;
          border-radius: 10px;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 6px;
          transition: background-color 0.15s ease, border-color 0.15s ease;
          box-sizing: border-box;
        }}
        .copy-btn:hover {{
          background-color: #F5F3FF;
          border-color: #C4B5FD;
        }}
        .copy-btn.copied {{
          background-color: #F3EEFF;
          border-color: #7C3AED;
          color: #5B21B6;
        }}
      </style>
    </head>
    <body>
      <button id="copyStoryBtn" class="copy-btn" type="button" onclick="copyStoryText()">
        Copy Story
      </button>
      <script>
        const storyPayload = {safe_json_text};
        function showCopiedState() {{
          const btn = document.getElementById("copyStoryBtn");
          btn.textContent = "Copied!";
          btn.classList.add("copied");
          setTimeout(() => {{
            btn.textContent = "Copy Story";
            btn.classList.remove("copied");
          }}, 1800);
        }}
        function fallbackCopy(text) {{
          const area = document.createElement("textarea");
          area.value = text;
          area.style.position = "fixed";
          area.style.opacity = "0";
          document.body.appendChild(area);
          area.focus();
          area.select();
          try {{
            document.execCommand("copy");
            showCopiedState();
          }} finally {{
            document.body.removeChild(area);
          }}
        }}
        function copyStoryText() {{
          if (navigator.clipboard && navigator.clipboard.writeText) {{
            navigator.clipboard.writeText(storyPayload)
              .then(showCopiedState)
              .catch(() => fallbackCopy(storyPayload));
          }} else {{
            fallbackCopy(storyPayload);
          }}
        }}
      </script>
    </body>
    </html>
    """
    components.html(button_html, height=44)


def _render_input_card() -> None:
    """Render the prompt input, controls, and Generate Continuation action."""
    with st.container(border=True):
        st.text_area(
            "Story Opening / Prompt",
            key="story_prompt",
            height=145,
            placeholder=(
                "Write the opening lines of your story...\n"
                "Example: On a quiet autumn evening, Maya discovered a brass key tucked inside an old library book."
            ),
        )

        col_length, col_temp = st.columns([1, 1.35], gap="medium")
        with col_length:
            selected_length = st.selectbox(
                "Response Length",
                options=LENGTH_OPTIONS,
                key="response_length",
                help="Choose how long the generated continuation should be.",
            )
            st.caption(f"Target: {LENGTH_PRESETS[selected_length]['approx_words']}")

        with col_temp:
            st.slider(
                "Creativity (Temperature)",
                min_value=MIN_TEMPERATURE,
                max_value=MAX_TEMPERATURE,
                step=0.1,
                key="temperature",
                help="Lower values keep the continuation focused; higher values encourage bolder twists.",
            )
            st.caption("0.1 = Focused  ·  0.8 = Balanced  ·  1.5 = Creative")

        with st.expander("Inference Backend Settings", expanded=False):
            st.selectbox(
                "Inference Backend",
                options=BACKEND_OPTIONS,
                key="inference_backend",
                help="Use TorchServe REST API (with optional local DistilGPT2 fallback) or run locally.",
            )
            st.text_input(
                "TorchServe Prediction URL",
                key="torchserve_url",
                help="REST endpoint exposed by TorchServe (default: http://localhost:8090/predictions/story_continuation).",
            )

        if st.session_state.get("validation_error"):
            st.warning(st.session_state["validation_error"])

        if st.session_state.get("backend_notice"):
            st.info(st.session_state["backend_notice"])

        if st.session_state.get("generation_error"):
            st.error(st.session_state["generation_error"])

        col_generate, col_reset = st.columns([3, 1], gap="small")
        with col_generate:
            generate_clicked = st.button(
                "Generate Continuation",
                type="primary",
                use_container_width=True,
            )
        with col_reset:
            st.button(
                "Reset",
                type="secondary",
                use_container_width=True,
                on_click=_on_reset_click,
            )

        if generate_clicked:
            is_valid, validation_msg = validate_story_prompt(st.session_state["story_prompt"])
            if not is_valid:
                st.session_state["validation_error"] = validation_msg
                st.session_state["generation_error"] = None
                st.session_state["backend_notice"] = None
                st.rerun()

            backend_choice = st.session_state["inference_backend"]
            spinner_label = (
                "Generating story continuation via TorchServe API..."
                if backend_choice != BACKEND_LOCAL
                else "Generating story continuation with local DistilGPT2..."
            )
            with st.spinner(spinner_label):
                result, error_msg = handle_generate_request(
                    prompt=st.session_state["story_prompt"],
                    length_option=st.session_state["response_length"],
                    temperature=st.session_state["temperature"],
                    backend=backend_choice,
                    torchserve_url=st.session_state["torchserve_url"],
                )

            st.session_state["validation_error"] = None
            if result is not None:
                st.session_state["generated_result"] = result
                st.session_state["generation_error"] = None
                if result.get("fallback_reason"):
                    st.session_state["backend_notice"] = (
                        "TorchServe API was unreachable; generated using local DistilGPT2 fallback."
                    )
                else:
                    st.session_state["backend_notice"] = None
            else:
                st.session_state["generation_error"] = error_msg
                st.session_state["backend_notice"] = None
            st.rerun()


def _render_output_card() -> None:
    """Render the Generated Story output card and Copy / Reset actions."""
    result: dict[str, Any] | None = st.session_state.get("generated_result")

    with st.container(border=True):
        if result is None:
            st.markdown(
                """
                <div class="story-card-header">
                    <h2 class="story-card-title">Generated Story</h2>
                    <span class="story-pill">DistilGPT2</span>
                </div>
                <div class="story-empty-state">
                    Your continued story will appear here.<br/>
                    Enter a story opening above and click <strong>Generate Continuation</strong> to begin.
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        escaped_prompt = html.escape(result["prompt"])
        escaped_continuation = html.escape(result["continuation"])
        backend_label = html.escape(str(result.get("backend", "DistilGPT2 · CPU")))
        if not result["continuation"]:
            separator = ""
        elif result["continuation"][0] in (",", ".", "!", "?", ";", ":", "'", '"', "”", ")"):
            separator = ""
        else:
            separator = " "

        st.markdown(
            f"""
            <div class="story-card-header">
                <h2 class="story-card-title">Generated Story</h2>
                <div class="story-meta-group">
                    <span class="story-pill">{html.escape(result['length_option'])}</span>
                    <span class="story-pill">Creativity {result['temperature']:.1f}</span>
                    <span class="story-pill">{backend_label}</span>
                </div>
            </div>
            <div class="story-body"><span class="story-prompt-lead">{escaped_prompt}</span>{separator}{escaped_continuation}</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        col_copy, col_reset = st.columns([1, 1], gap="small")
        with col_copy:
            _render_copy_button(result["full_story"])
        with col_reset:
            st.button(
                "Reset Story",
                key="reset_output_btn",
                type="secondary",
                use_container_width=True,
                on_click=_on_reset_click,
            )


def _render_disclaimer() -> None:
    """Render the AI-generated content disclaimer at the bottom of the page."""
    st.markdown(
        """
        <div class="ai-disclaimer">
            <strong>Disclaimer:</strong> Content shown in this application is AI-generated using the
            pretrained DistilGPT2 language model. Generated stories are purely fictional and may not
            reflect factual information.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Configure and render the AI Story Continuation System Streamlit application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📖",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    initialize_session_state(st.session_state)
    _inject_custom_styles()
    _render_header()
    _render_input_card()
    _render_output_card()
    _render_disclaimer()


if __name__ == "__main__":
    main()
