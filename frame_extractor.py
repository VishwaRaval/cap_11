import cv2, math, os, glob

# Use a raw string for Windows paths to avoid escape issues
INPUT_DIR = r"D:\New York University\Third Semester\Capstone Project\Videos"
OUT_ROOT  = "frames"     # output base folder
FPS_STEP  = 1            # seconds between frames

os.makedirs(OUT_ROOT, exist_ok=True)

for fp in glob.glob(os.path.join(INPUT_DIR, "*.mp4")):
    cap = cv2.VideoCapture(fp)
    if not cap.isOpened():
        print(f"[warn] cannot open {fp}")
        continue

    bn = os.path.splitext(os.path.basename(fp))[0]
    out_dir = os.path.join(OUT_ROOT, bn)
    os.makedirs(out_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if fps and fps > 0 else 0
    n_shots = int(math.floor(duration_s)) + 1

    saved = 0
    for s in range(0, n_shots, FPS_STEP):
        # seek by timestamp (ms)
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # --- remove alpha channel using cvtColor when present ---
        # Note: frames decoded by OpenCV are typically BGR (3 ch). If 4 ch, they are BGRA.
        if frame.ndim == 3 and frame.shape[2] == 4:
            # Prefer BGRA -> BGR for OpenCV ordering
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # If you specifically want RGBA->RGB instead, use:
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.ndim == 2:
            # grayscale → BGR (3 ch) to ensure 24-bit PNG
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        out_path = os.path.join(out_dir, f"{bn}_{s:05d}.png")
        cv2.imwrite(out_path, frame)  # PNG saved as 24-bit (no alpha)
        saved += 1

    cap.release()
    print(f"[done] {bn}: saved {saved} frames to {out_dir}")
