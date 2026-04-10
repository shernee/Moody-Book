# StoryScore — Setup & Run

## Prerequisites

### 1. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1
```

### 2. Python dependencies
```bash
pip install fastapi uvicorn audiocraft torch torchaudio requests
```
> audiocraft installs MusicGen. On first run it downloads the model weights (~1.5GB for 'small').

---

## Running

### Step 1 — Start Ollama (in a terminal)
```bash
ollama serve
```

### Step 2 — Start the API (in another terminal)
```bash
cd storyscore/backend
uvicorn api:app --reload --port 8000
```

### Step 3 — Open the frontend
Open `storyscore/frontend/index.html` in your browser directly,
or serve it:
```bash
cd storyscore/frontend
python -m http.server 3000
# then open http://localhost:3000
```

---

## Usage

1. **Setup tab** — paste your book pages (or click "Load Peter Rabbit" to demo)
2. Click **Analyse Book** — Ollama analyses each page's mood (~20s for 18 pages)
3. Click **Generate Audio** — MusicGen generates one 30s clip per page (runs in background)
   - CPU: ~3 mins per clip (plan for an hour+ for a full book)
   - GPU: ~15s per clip
4. Click **Read** — opens the reader. Tap Next/Prev or set auto-advance timer.

---

## Generating audio offline (CLI, no UI needed)
```bash
cd storyscore/backend

# Step 1: analyse
python analyser.py
# creates output/the_tale_of_peter_rabbit_profile.json

# Step 2: generate
python generator.py output/the_tale_of_peter_rabbit_profile.json
# creates output/audio/the_tale_of_peter_rabbit/page_01.wav etc
```

---

## Tuning audio quality

Edit `generator.py`:
```python
MODEL_SIZE = "small"    # fast, decent
MODEL_SIZE = "medium"   # slower, noticeably better
CLIP_DURATION = 30      # seconds per clip
```

Edit the `CHILDREN_STYLE_SUFFIX` in `generator.py` to change the music style globally.

---

## Project structure
```
storyscore/
├── backend/
│   ├── analyser.py      # Ollama mood analysis
│   ├── generator.py     # MusicGen audio generation
│   └── api.py           # FastAPI backend
├── books/
│   └── peter_rabbit.py  # Book data (add more here)
├── frontend/
│   └── index.html       # Single-file UI
└── output/
    ├── *_profile.json   # Mood profiles
    └── audio/           # Generated .wav files
```
