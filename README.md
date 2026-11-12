# AI Story Continuation System (Phase 3: TorchServe Integration)

An interactive NLP web application built with **Streamlit**, **PyTorch**, **Hugging Face Transformers**, and **TorchServe** (`0.11.0`) that serves pretrained **`distilgpt2`** story continuations over a REST API (with automatic local DistilGPT2 fallback).

> **Phase 3 Scope:** Custom TorchServe handler (`torchserve/handler.py`), `.mar` model archive packaging (`model_store/story_continuation.mar`), isolated Linux deployment configuration (`Dockerfile` + WSL2 Ubuntu runtime), REST API verification, and Streamlit TorchServe integration. AWS SageMaker deployment remains deferred to a future phase.

---

## TorchServe Architecture

```text
Streamlit UI (Windows venv, localhost:8501)
   │  HTTP POST JSON {"prompt", "length", "temperature"}
   ▼
TorchServe REST API (Linux WSL2 / Docker, localhost:8090)
   ├── Inference API  : http://localhost:8090/predictions/story_continuation
   ├── Management API : http://localhost:8091/models
   └── Metrics API    : http://localhost:8092/metrics
   │
   ▼
story_continuation.mar
   ├── torchserve/handler.py (StoryContinuationHandler)
   └── Packaged DistilGPT2 weights & tokenizer (loaded once from model_dir)
```

*(Ports `8090`/`8091`/`8092` are configured in `torchserve/config.properties` to avoid conflicts with services commonly bound to port `8080` on Windows, and the Streamlit UI lets you override the endpoint URL or switch to `Local DistilGPT2` anytime.)*

---

## Custom Handler Explanation (`torchserve/handler.py`)

`StoryContinuationHandler` implements the standard TorchServe lifecycle:
1. **`initialize(context)`**: Resolves packaged DistilGPT2 files directly from `context.system_properties["model_dir"]` (avoiding repeated downloads during inference), sets `tokenizer.pad_token = tokenizer.eos_token`, moves the model to CPU, and switches to `model.eval()`.
2. **`preprocess(data)`**: Parses UTF-8 JSON bodies, validates required `"prompt"` (non-empty string), `"length"` (`Short` = `50`, `Medium` = `100`, `Long` = `180`), and `"temperature"` (`0.1` to `1.5`), and tokenizes with truncation (`max_length = 1024 - max_new_tokens`) so very long prompts never exceed DistilGPT2's context window. Invalid inputs raise `PredictionException(..., 400)`.
3. **`inference(model_input)`**: Runs `model.generate(...)` inside `torch.inference_mode()` with `do_sample=True`, `top_p=0.92`, `repetition_penalty=1.15`, and `no_repeat_ngram_size=3`, slicing only newly generated tokens (`output_ids[0][prompt_len:]`).
4. **`postprocess(inference_output)`**: Returns structured JSON responses matching the API schema.

---

## Model Packaging Process

[torchserve/package_model.py](torchserve/package_model.py) saves `distilgpt2` tokenizer & `model.safetensors` locally and builds `model_store/story_continuation.mar` using `torch-model-archiver`:

```bash
python torchserve/package_model.py
```

This bundles `torchserve/handler.py`, `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, and `special_tokens_map.json` into `model_store/story_continuation.mar` (~291.6 MB).

---

## API Request & Response Formats

**Endpoint:** `POST http://localhost:8090/predictions/story_continuation`

### Request JSON
```json
{
  "prompt": "One day, a student discovered a mysterious box in the college library...",
  "length": "Medium",
  "temperature": 0.8
}
```
- `prompt` (required, non-empty string)
- `length` (optional, default `"Medium"`): `"Short"` (`50` tokens), `"Medium"` (`100` tokens), `"Long"` (`180` tokens)
- `temperature` (optional, default `0.8`): float in `[0.1, 1.5]`

### Success Response JSON (`200 OK`)
```json
{
  "generated_text": "and it was buried at The University of Texas-Austin...",
  "length": "Medium",
  "temperature": 0.8
}
```

### Validation Error JSON (`400 Bad Request`)
```json
{
  "code": 400,
  "type": "InternalServerException",
  "message": "Field 'prompt' must be a non-empty string."
}
```

---

## Local Deployment Instructions (Windows 11)

TorchServe requires a Linux runtime. Two isolated options are provided (neither alters your Windows `venv`):

### Option A: WSL2 Ubuntu (Verified on this machine)
Run TorchServe inside an isolated user-space Linux environment (`~/ts310` with Python `3.10.11`, OpenJDK `17`, `torch==2.2.2+cpu`, `transformers==4.44.2`, `numpy==1.26.4`, `torchserve==0.11.0`):

```bash
export PATH="$HOME/ts310/bin:$PATH"
export LOG_LOCATION=/tmp/ts_logs
export METRICS_LOCATION=/tmp/ts_logs
mkdir -p /tmp/ts_logs
cd /mnt/c/Users/Hp/OneDrive/Desktop/AI-Story-Continuation
torchserve --start --ncs \
  --model-store /mnt/c/Users/Hp/OneDrive/Desktop/AI-Story-Continuation/model_store \
  --models story_continuation=story_continuation.mar \
  --ts-config /mnt/c/Users/Hp/OneDrive/Desktop/AI-Story-Continuation/torchserve/config.properties \
  --foreground
```

### Option B: Docker Linux Container (`Dockerfile`)
Once Docker Desktop is running:

```powershell
docker build -t ai-story-torchserve .
docker run --rm -p 8090:8090 -p 8091:8091 -p 8092:8092 ai-story-torchserve
```

### Launch Streamlit Frontend (Windows PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

In the UI, expand **Inference Backend Settings** to switch between:
- `TorchServe API (Auto Fallback)` *(default — calls TorchServe and falls back to local DistilGPT2 if unreachable)*
- `TorchServe API` *(strict REST API mode)*
- `Local DistilGPT2` *(direct in-process inference)*

---

## API & Unit Testing Commands

### Health & Prediction `curl` Tests
```powershell
curl.exe http://localhost:8090/ping
curl.exe http://localhost:8091/models
curl.exe -X POST http://localhost:8090/predictions/story_continuation ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"One day, a student discovered a mysterious box in the college library...\",\"length\":\"Medium\",\"temperature\":0.8}"
```

### Offline Unit Tests
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```
