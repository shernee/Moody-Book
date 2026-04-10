"""
api.py
------
FastAPI backend exposing the analyser and generator as HTTP endpoints.

Run:  uvicorn api:app --reload --port 8000
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from analyser import analyse_book, save_profile, load_profile, BookProfile
from generator import generate_book_audio, OUTPUT_DIR

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

def run_generation(profile: BookProfile):
    title = profile.title
    generation_status[title] = "generating"
    try:
        generate_book_audio(profile)
        generation_status[title] = "complete"
    except Exception as e:
        generation_status[title] = f"error: {str(e)}"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/analyse", response_model=BookResponse)
def analyse(book: BookInput):
    """Analyse a book and return mood profiles for each page."""
    try:
        profile = analyse_book(book.title, book.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    os.makedirs("output", exist_ok=True)
    save_profile(profile, str(profile_path(book.title)))
    generation_status[book.title] = "analysed"
    
    return book_to_response(profile)

@app.post("/generate/{title}")
def generate(title: str, background_tasks: BackgroundTasks):
    """Trigger audio generation for a previously analysed book."""
    path = profile_path(title)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Book not analysed yet. Run /analyse first.")
    
    if generation_status.get(title) == "generating":
        return {"status": "already generating", "title": title}
    
    profile = load_profile(str(path))
    background_tasks.add_task(run_generation, profile)
    generation_status[title] = "generating"
    
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
    """List all analysed books."""
    output = Path("output")
    books = []
    for f in output.glob("*_profile.json"):
        title_slug = f.stem.replace("_profile", "")
        books.append({
            "slug": title_slug,
            "status": generation_status.get(title_slug.replace("_", " ").title(), "unknown")
        })
    return {"books": books}
