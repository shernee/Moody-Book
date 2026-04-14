"""
analyser.py
-----------
Takes a children's book (list of pages) and returns a mood/instrumentation
profile for each page, ready to feed into the music generator.

Requires Ollama running locally:  ollama run llama3.1
"""

import json
import os
import re
import requests
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PageProfile:
    page: int
    text: str
    mood: str           # e.g. "warm and cozy", "tense and suspenseful"
    energy: str         # "low" | "medium" | "high"
    tempo: str          # "slow" | "moderate" | "lively"
    instrumentation: str  # e.g. "soft piano, gentle strings"
    music_prompt: str   # final prompt ready for MusicGen

@dataclass
class BookProfile:
    title: str
    pages: List[PageProfile]

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a music supervisor for children's audiobooks.
Given a short passage from a children's picture book, you analyse the emotional 
tone and suggest background music characteristics.

You must respond ONLY with a valid JSON object — no explanation, no markdown, 
no backticks. Just the raw JSON.

The JSON must have exactly these fields:
{
  "mood": "short phrase describing the emotional tone",
  "energy": "low | medium | high",
  "tempo": "slow | moderate | lively",
  "instrumentation": "2-4 instruments that would suit this moment",
  "music_prompt": "a single descriptive sentence for a music generation model"
}

Guidelines for children's book music:
- Keep it gentle and age-appropriate (toddler/preschool)
- music_prompt should mention: tempo, mood, instruments, and style
- Avoid anything dark, scary, or overly dramatic
- Think: bedtime stories, nursery rhymes, soft lullabies, playful jingles
"""

def build_user_message(page_text: str) -> str:
    return f"Analyse this children's book page and return the JSON profile:\n\n\"{page_text}\""

# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.3,   # low temp for consistent structured output
            "top_p": 0.9,
            "num_predict": 512,
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"]

def parse_response(raw: str) -> dict:
    """Extract JSON from model response robustly."""
    # Strip markdown fences if the model adds them despite instructions
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    
    # Find first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response:\n{raw}")
    
    data = json.loads(match.group())
    # Normalise any list fields the model may return as arrays
    for key in ("instrumentation", "mood", "energy", "tempo", "music_prompt"):
        if key in data and isinstance(data[key], list):
            data[key] = ", ".join(data[key])
    return data

# ── Main analyser ─────────────────────────────────────────────────────────────

def analyse_page(page_num: int, text: str, retries: int = 3) -> PageProfile:
    """Analyse a single page and return its PageProfile."""
    print(f"  Analysing page {page_num}...", end=" ", flush=True)

    last_error = None
    for attempt in range(retries):
        try:
            raw = call_ollama(build_user_message(text))
            data = parse_response(raw)
            profile = PageProfile(
                page=page_num,
                text=text,
                mood=data["mood"],
                energy=data["energy"],
                tempo=data["tempo"],
                instrumentation=data["instrumentation"],
                music_prompt=data["music_prompt"],
            )
            print(f"✓ [{profile.mood}]")
            return profile
        except Exception as e:
            last_error = e
            print(f"\n  ⚠️  Attempt {attempt + 1} failed: {e}. Retrying...")

    raise ValueError(f"Failed to analyse page {page_num} after {retries} attempts: {last_error}")

def analyse_book(title: str, pages: List[str], force_reanalyse: bool = False) -> BookProfile:
    """Analyse all pages of a book and return a BookProfile.

    If a saved profile already exists and force_reanalyse is False, the cached
    profile is loaded and returned immediately without calling Ollama.
    """
    output_path = Path("output") / f"{title.lower().replace(' ', '_')}_profile.json"
    if not force_reanalyse and output_path.exists():
        print(f"\n📖 Loading cached profile for: {title}")
        return load_profile(output_path)

    print(f"\n📖 Analysing: {title}")
    print(f"   {len(pages)} pages to process\n")
    
    profiles = []
    for i, text in enumerate(pages, start=1):
        profile = analyse_page(i, text)
        profiles.append(profile)
    
    print(f"\n✅ Analysis complete for '{title}'")
    return BookProfile(title=title, pages=profiles)

def save_profile(book_profile: BookProfile, output_path: str):
    """Save the book profile to a JSON file."""
    data = {
        "title": book_profile.title,
        "pages": [asdict(p) for p in book_profile.pages]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Profile saved to {output_path}")

def load_profile(path: str) -> BookProfile:
    """Load a previously saved book profile."""
    with open(path) as f:
        data = json.load(f)
    pages = [PageProfile(**p) for p in data["pages"]]
    return BookProfile(title=data["title"], pages=pages)

# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from books.peter_rabbit import TITLE, PAGES
    
    profile = analyse_book(TITLE, PAGES)
    
    output_path = f"output/{TITLE.lower().replace(' ', '_')}_profile.json"
    os.makedirs("output", exist_ok=True)
    save_profile(profile, output_path)
    
    # Pretty print summary
    print("\n── Page Summary ──────────────────────────────")
    for p in profile.pages:
        print(f"  Page {p.page:2d} | {p.tempo:10s} | {p.mood}")
