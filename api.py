"""
api.py
------
FastAPI backend exposing the analyser and generator as HTTP endpoints.

Run:  uvicorn api:app --reload --port 8000
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from analyser import analyse_book, save_profile, load_profile, BookProfile
from generator import generate_book_audio, regenerate_page, OUTPUT_DIR

app = FastAPI(title="StoryScore API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated audio files statically
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(OUTPUT_DIR)), name="audio")

# ── Request / Response models ─────────────────────────────────────────────────

class BookInput(BaseModel):
    title: str
    pages: List[str]

class PageProfileResponse(BaseModel):
    page: int
    text: str
    mood: str
    energy: str
    tempo: str
    instrumentation: str
    music_prompt: str
    audio_url: Optional[str] = None

class BookResponse(BaseModel):
    title: str
    status: str
    pages: List[PageProfileResponse]

# ── State (in-memory for now) ─────────────────────────────────────────────────

# Maps book title → generation status
generation_status: dict[str, str] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────

def profile_path(title: str) -> Path:
    return Path(f"output/{title.lower().replace(' ', '_')}_profile.json")

def audio_path(title: str, page: int) -> Optional[str]:
    book_dir = OUTPUT_DIR / title.lower().replace(" ", "_")
    wav = book_dir / f"page_{page:02d}.wav"
    if wav.exists():
        return f"/audio/{title.lower().replace(' ', '_')}/page_{page:02d}.wav"
    return None

def book_to_response(profile: BookProfile) -> BookResponse:
    title = profile.title
    pages = [
        PageProfileResponse(
            **{k: v for k, v in vars(p).items()},
            audio_url=audio_path(title, p.page)
        )
        for p in profile.pages
    ]
    return BookResponse(
        title=title,
        status=generation_status.get(title, "ready"),
        pages=pages
    )

# ── Background task ───────────────────────────────────────────────────────────

def run_generation(profile: BookProfile, force_regenerate: bool = False):
    title = profile.title
    generation_status[title] = "generating"
    try:
        generate_book_audio(profile, force_regenerate=force_regenerate)
        generation_status[title] = "complete"
    except Exception as e:
        generation_status[title] = f"error: {str(e)}"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/analyse", response_model=BookResponse)
def analyse(book: BookInput, force: bool = False):
    """Analyse a book and return mood profiles for each page.

    If a profile already exists and ?force=true is not set, the cached profile
    is returned immediately without calling Ollama.
    """
    try:
        profile = analyse_book(book.title, book.pages, force_reanalyse=force)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    os.makedirs("output", exist_ok=True)
    save_profile(profile, str(profile_path(book.title)))
    generation_status[book.title] = "analysed"

    return book_to_response(profile)

@app.post("/generate/{title}")
def generate(title: str, background_tasks: BackgroundTasks, force_regenerate: bool = False):
    """Trigger audio generation for a previously analysed book.

    By default, pages that already have a .wav file are skipped.
    Pass force_regenerate=true to overwrite all existing audio.
    """
    path = profile_path(title)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Book not analysed yet. Run /analyse first.")

    profile = load_profile(str(path))

    if generation_status.get(profile.title) == "generating":
        return {"status": "already generating", "title": title}

    background_tasks.add_task(run_generation, profile, force_regenerate)
    generation_status[profile.title] = "generating"

    return {"status": "generation started", "title": title}

@app.get("/status/{title}")
def status(title: str):
    """Check generation status for a book."""
    return {
        "title": title,
        "status": generation_status.get(title, "not found"),
        "audio_ready": generation_status.get(title) == "complete"
    }

@app.get("/book/{title}", response_model=BookResponse)
def get_book(title: str):
    """Get full profile and audio URLs for a book."""
    path = profile_path(title)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Book not found.")
    profile = load_profile(str(path))
    return book_to_response(profile)

@app.get("/books")
def list_books():
    """List all analysed books with profile and audio status."""
    output = Path("output")
    books = []
    for f in sorted(output.glob("*_profile.json")):
        slug = f.stem.replace("_profile", "")
        try:
            profile = load_profile(str(f))
        except Exception:
            continue

        page_count = len(profile.pages)
        audio_dir = OUTPUT_DIR / slug
        audio_pages = len(list(audio_dir.glob("page_*.wav"))) if audio_dir.exists() else 0

        # Prefer in-memory status (accurate during active generation); fall back to file state
        mem_status = generation_status.get(profile.title)
        if mem_status:
            status = mem_status
        elif audio_pages == 0:
            status = "analysed"
        elif audio_pages < page_count:
            status = "partial"
        else:
            status = "complete"

        books.append({
            "title": profile.title,
            "slug": slug,
            "page_count": page_count,
            "audio_pages": audio_pages,
            "status": status,
        })
    return {"books": books}


@app.delete("/book/{title}/profile")
def delete_profile(title: str):
    """Delete the analysis profile for a book (forces re-analyse on next request)."""
    path = profile_path(title)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found.")
    # Load to get the real title so we can clear the correct status key
    try:
        profile = load_profile(str(path))
        generation_status.pop(profile.title, None)
    except Exception:
        pass
    path.unlink()
    generation_status.pop(title, None)
    return {"deleted": str(path)}


@app.delete("/book/{title}/audio")
def delete_audio(title: str):
    """Delete all generated audio for a book."""
    slug = title.lower().replace(" ", "_")
    audio_dir = OUTPUT_DIR / slug
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    # Clear status so generation can be restarted
    try:
        path = profile_path(title)
        if path.exists():
            profile = load_profile(str(path))
            generation_status.pop(profile.title, None)
    except Exception:
        pass
    generation_status.pop(title, None)
    return {"deleted": str(audio_dir)}


@app.delete("/book/{title}/audio/{page_num}")
def delete_page_audio(title: str, page_num: int):
    """Delete the audio file for a single page."""
    slug = title.lower().replace(" ", "_")
    wav = OUTPUT_DIR / slug / f"page_{page_num:02d}.wav"
    if not wav.exists():
        raise HTTPException(status_code=404, detail=f"Audio for page {page_num} not found.")
    wav.unlink()
    return {"deleted": str(wav)}
