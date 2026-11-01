"""Rebuild every asset under static/ from the paper and capture sources (local only).

Run from anywhere: python3 tools/build_assets.py
Needs pdftocairo, ImageMagick `convert`, and OpenCV (VP9 WebM writer).
"""
import json
import shutil
import subprocess
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
IMG, VID = ROOT / "static/img", ROOT / "static/video"
PAPER = Path("/home/cislab301b/Aditya/Writing/paper/figures")
CAP = PAPER / "isaac_captures"
PLAY = Path("/home/cislab301b/Aditya/DualHeadPPO/play_outputs")

FIGS = {"fig1": "Fig1_fixed", "fig2": "Fig2", "fig3": "fig3_eval_error",
        "fig4": "fig4_training", "fig5": "fig5_robustness"}

# Ablation 15 seed 42 evaluation runs with wrist-camera recordings
WRIST = {
    "wrist_divc": PLAY / "2026-10-19_11-31-06_PPO_single_sym_decomposed_policy_2026-10-31_08-05-02/camera_det_2026-10-31_08-05-02.mp4",
    "wrist_ppo": PLAY / "2026-10-19_10-17-42_PPO_single_sym_policy_2026-10-31_08-08-01/camera_det_2026-10-31_08-08-01.mp4",
}

# Same placement (nominal ep0), fixed camera az60, one frame every 3 policy steps (0.1 s at 30 Hz)
SEQ = {"divc": ("divc_nominal_seq_s42", 47), "ppo": ("ppo_nominal_seq_s42", 65)}
CROP = (slice(40, 1080), slice(700, 1856))  # rows, cols: arm + target, drops empty floor
SEQ_SIZE = (720, 648)
FPS = 10
HOLD_S = 1.2  # pause on the positioned frame before the loop restarts


def run(*cmd):
    subprocess.run(cmd, check=True)


def figures():
    for name, src in FIGS.items():
        png = IMG / f"{name}.png"
        run("pdftocairo", "-png", "-singlefile", "-scale-to-x", "2000", "-scale-to-y", "-1", str(PAPER / f"{src}.pdf"), str(png.with_suffix("")))
        run("convert", str(png), "-quality", "88", str(IMG / f"{name}.webp"))
        png.unlink()


def sequences():
    hero = {}
    for key, (stem, t_end) in SEQ.items():
        frames = [CAP / f"D_sequence_filmstrip/{stem}__ep00_t{t:03d}_{lab}_fixed_az60.png"
                  for t, lab in [(t, "start" if t == 0 else "seq") for t in range(0, t_end, 3)]]
        frames.append(CAP / f"D_sequence_filmstrip/{stem}__ep00_t{t_end:03d}_positioned_fixed_az60.png")
        hero[key] = {"t_end": t_end, "frames": len(frames)}
        out = cv2.VideoWriter(str(VID / f"seq_{key}.webm"), cv2.VideoWriter_fourcc(*"VP90"), FPS, SEQ_SIZE)
        assert out.isOpened(), "VP9 writer unavailable"
        imgs = [cv2.resize(cv2.imread(str(f))[CROP], SEQ_SIZE, interpolation=cv2.INTER_AREA) for f in frames]
        # pad both clips to the PPO length so the two videos stay in step
        n_total = (SEQ["ppo"][1] // 3 + 1) + int(HOLD_S * FPS)
        for i in range(n_total):
            out.write(imgs[min(i, len(imgs) - 1)])
        out.release()
        cv2.imwrite(str(IMG / f"seq_{key}_poster.webp"), imgs[0], [cv2.IMWRITE_WEBP_QUALITY, 80])

    # per-axis error (mm) for the hero trace, straight from the capture logs
    for key, (stem, _) in SEQ.items():
        log = json.loads((CAP / f"logs/{stem}_capture_log.json").read_text())
        rows = [r for r in log if r["ep"] == 0 and r["label"] in ("start", "seq", "positioned")]
        hero[key]["t"] = [r["t"] for r in rows]
        hero[key]["err"] = [[round(abs(v), 3) for v in r["err_mm"]] for r in rows]
    hero["fps"], hero["policy_hz"], hero["stride"] = FPS, 30, 3
    (ROOT / "static/hero.json").write_text(json.dumps(hero, separators=(",", ":")))


def wrist():
    for name, src in WRIST.items():
        shutil.copyfile(src, VID / f"{name}.mp4")  # already H.264, 640x640, 30 fps
        cap = cv2.VideoCapture(str(src))
        ok, frame = cap.read()
        assert ok, src
        cv2.imwrite(str(IMG / f"{name}_poster.webp"), frame, [cv2.IMWRITE_WEBP_QUALITY, 80])


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    figures()
    sequences()
    wrist()
    print("done")
