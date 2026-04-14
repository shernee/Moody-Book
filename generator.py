"""
generator.py
------------
Takes a BookProfile (from analyser.py) and generates one audio clip per page
using Meta's MusicGen model via HuggingFace transformers.

Install: pip install -r requirements.txt
Model downloads automatically on first run (~1.5GB for 'small').

Note on performance:
  - With GPU: ~10-15s per 30s clip
  - CPU only: ~2-4 mins per 30s clip
  Generation is done offline and upfront — not in real time during reading.
"""

import os
import torch
import scipy.io.wavfile
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from analyser import BookProfile, load_profile

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_SIZE = "small"       # "small" (300M) | "medium" (1.5B) | "large" (3.3B)
CLIP_DURATION = 15         # seconds per page clip
OUTPUT_DIR = Path("output/audio")

MODEL_IDS = {
    "small":  "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large":  "facebook/musicgen-large",
}

# ── Model loader (singleton) ──────────────────────────────────────────────────

_model = None
_processor = None

def get_model():
    global _model, _processor
    if _model is None:
        model_id = MODEL_IDS[MODEL_SIZE]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎵 Loading MusicGen ({MODEL_SIZE}) on {device}...", end=" ", flush=True)

        _processor = AutoProcessor.from_pretrained(model_id)
        _model = MusicgenForConditionalGeneration.from_pretrained(model_id)
        _model.to(device)

        print("✓")
    return _model, _processor

# ── Style wrapper ─────────────────────────────────────────────────────────────

CHILDREN_STYLE_SUFFIX = (
    ", children's music, gentle, soft, toddler-friendly, soothing, "
    "high quality audio, no vocals, no lyrics"
)

def build_music_prompt(page_prompt: str) -> str:
    return page_prompt + CHILDREN_STYLE_SUFFIX

# ── Generator ─────────────────────────────────────────────────────────────────

def generate_clip(prompt: str, output_path: Path) -> Path:
    """Generate a single audio clip and save as .wav. Returns the output path."""
    model, processor = get_model()
    device = next(model.parameters()).device

    full_prompt = build_music_prompt(prompt)

    inputs = processor(
        text=[full_prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # tokens_per_second ≈ 50 for MusicGen
    max_new_tokens = CLIP_DURATION * 50

    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # audio_values shape: [batch, channels, samples]
    audio_np = audio_values[0, 0].cpu().numpy()

    # Normalise to int16
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    sample_rate = model.config.audio_encoder.sampling_rate

    output_path = output_path.with_suffix(".wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(str(output_path), sample_rate, audio_int16)

    return output_path

def generate_book_audio(book_profile: BookProfile, output_dir: Path = OUTPUT_DIR, force_regenerate: bool = False):
    """Generate one audio clip per page for an entire book.

    If force_regenerate is False (default), pages that already have a .wav file
    are skipped — only missing clips are generated.
    """
    book_dir = output_dir / book_profile.title.lower().replace(" ", "_")
    book_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎼 Generating audio for: {book_profile.title}")
    print(f"   {len(book_profile.pages)} clips × {CLIP_DURATION}s each (looping)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        estimated = len(book_profile.pages) * 3
        print(f"   ⚠️  CPU mode — estimated {estimated} mins total")

    print()

    generated = []
    for page in book_profile.pages:
        clip_path = book_dir / f"page_{page.page:02d}"
        wav_path = clip_path.with_suffix(".wav")

        if not force_regenerate and wav_path.exists():
            print(f"  Page {page.page:2d} | skipping (already exists)")
            generated.append(str(wav_path))
            continue

        print(f"  Page {page.page:2d} | {page.mood}")
        print(f"          Prompt: {page.music_prompt[:70]}...")

        output = generate_clip(page.music_prompt, clip_path)
        generated.append(str(output))
        print(f"          ✓ Saved: {output.name}\n")

    print(f"✅ Audio generation complete.")
    print(f"   Files saved to: {book_dir}")
    return generated


def regenerate_page(profile: BookProfile, page_num: int, output_dir: Path = OUTPUT_DIR):
    """Regenerate audio for a single page, overwriting any existing file."""
    slug = profile.title.lower().replace(" ", "_")
    page_dir = output_dir / slug
    page_dir.mkdir(parents=True, exist_ok=True)

    page = next((p for p in profile.pages if p.page == page_num), None)
    if page is None:
        raise ValueError(f"Page {page_num} not found in profile for '{profile.title}'")

    clip_path = page_dir / f"page_{page_num:02d}"
    print(f"  ↺ Regenerating page {page_num} | {page.mood}")
    output = generate_clip(page.music_prompt, clip_path)
    print(f"    ✓ Saved: {output.name}")
    return str(output)

# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        profile_path = sys.argv[1]
    else:
        profile_path = "output/the_tale_of_peter_rabbit_profile.json"

    if not os.path.exists(profile_path):
        print(f"❌ Profile not found: {profile_path}")
        print("   Run analyser.py first to generate the profile.")
        sys.exit(1)

    profile = load_profile(profile_path)
    generate_book_audio(profile)
