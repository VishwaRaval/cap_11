from pathlib import Path
import shutil

# Auto-anchor to this script's directory (UNDP Capstone)
ROOT = Path(__file__).resolve().parent

DIR_FRAMES = ROOT / "frames"              # only fish images (in subfolders)
DIR_FRAMES_COPY = ROOT / "frames - Copy"  # all images (in subfolders)

OUT_FISH = ROOT / "fish"
OUT_NOFISH = ROOT / "no fish"           # per your request

# Image extensions to include
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def iter_images(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def main():
    # Safety: don’t recurse into output dirs if they already exist
    OUT_FISH.mkdir(parents=True, exist_ok=True)
    OUT_NOFISH.mkdir(parents=True, exist_ok=True)

    print("Roots:")
    print(f"  FRAMES:        {DIR_FRAMES}")
    print(f"  FRAMES - Copy: {DIR_FRAMES_COPY}")
    print(f"  OUT fish:      {OUT_FISH}")
    print(f"  OUT no fish:   {OUT_NOFISH}")

    # 1) Gather fish filenames (basenames are globally unique per your scheme)
    fish_names = {p.name for p in iter_images(DIR_FRAMES)}
    print(f"Found {len(fish_names)} fish filenames in 'frames'.")

    # 2) Copy all fish images from FRAMES to fish/
    copied_fish = 0
    for p in iter_images(DIR_FRAMES):
        dest = OUT_FISH / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
            copied_fish += 1

    # 3) For every image in FRAMES - Copy, if not in fish_names → no fish/
    copied_nofish = 0
    for p in iter_images(DIR_FRAMES_COPY):
        if p.name not in fish_names:
            dest = OUT_NOFISH / p.name
            if not dest.exists():
                shutil.copy2(p, dest)
                copied_nofish += 1

    # 4) Summary
    all_in_copy = sum(1 for _ in iter_images(DIR_FRAMES_COPY))
    all_in_frames = sum(1 for _ in iter_images(DIR_FRAMES))
    print("\n=== Done ===")
    print(f"Images in 'frames':        {all_in_frames}")
    print(f"Images in 'frames - Copy': {all_in_copy}")
    print(f"Copied to 'fish':          {copied_fish}")
    print(f"Copied to 'no fish':       {copied_nofish}")

if __name__ == "__main__":
    main()
