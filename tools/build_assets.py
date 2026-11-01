"""Rebuild every asset under static/ from the paper figures and the Isaac captures (local only).

Run: python3 tools/build_assets.py
Needs pdftocairo, ImageMagick `convert`, and OpenCV with a VP9 WebM writer.
All videos come from frames rendered by DualHeadPPO/tools/play_paper_capture.py
(Ablation 15, seed 42, deterministic policy). No new rollouts are made here.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG, VID = ROOT / "static/img", ROOT / "static/video"
PAPER = Path("/home/cislab301b/Aditya/Writing/paper/figures")
CAP = Path("/home/cislab301b/Aditya/DualHeadPPO/play_outputs/paper_capture")
POLICY_HZ = 30
GAP = 8  # white gutter between DivC-PPO and PPO halves
SQUARE_MACRO = ((0, 1080), (420, 1500))
SQUARE = {"closeup_side": ((0, 1080), (360, 1440))}  # square crop of 16:9 views around the gripper

FIGS = {"fig2": "Fig2", "fig3": "fig3_eval_error"}


def run(*cmd):
    subprocess.run(cmd, check=True)


def figures():
    for name, src in FIGS.items():
        png = IMG / f"{name}.png"
        run("pdftocairo", "-png", "-singlefile", "-scale-to-x", "2000", "-scale-to-y", "-1",
            str(PAPER / f"{src}.pdf"), str(png.with_suffix("")))
        run("convert", str(png), "-quality", "88", str(IMG / f"{name}.webp"))
        png.unlink()


def episode(run_dir, ep, last_t=None):
    """Rows of the capture log for one episode, up to last_t (the step after 209 is the reset)."""
    log = json.loads((CAP / run_dir / "capture_log.json").read_text())
    rows = [r for r in log if r["ep"] == ep and r["t"] <= 209]
    pos = [i for i, r in enumerate(rows) if r["label"] == "positioned"]
    if pos:  # after the first stable step the scripted grasp takes over
        rows = rows[:pos[0] + 1]
    if last_t is not None:
        rows = [r for r in rows if r["t"] <= last_t]
    return rows


def frame(run_dir, row, view, size, crop=None):
    path = CAP / run_dir / f"ep{row['ep']:02d}_t{row['t']:03d}_{row['label']}_{view}.png"
    img = cv2.imread(str(path))
    assert img is not None, path
    if crop:
        (y0, y1), (x0, x1) = crop
        img = img[y0:y1, x0:x1]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def write(name, frames, fps, poster=0):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(VID / f"{name}.webm"), cv2.VideoWriter_fourcc(*"VP90"), fps, (w, h))
    assert out.isOpened(), "VP9 writer unavailable"
    for f in frames:
        out.write(f)
    out.release()
    cv2.imwrite(str(IMG / f"{name}_poster.webp"), frames[poster], [cv2.IMWRITE_WEBP_QUALITY, 80])


def pair(name, left, right, view, size, fps, hold_s=1.2, crop=None, eps=(0, 0), last=(None, None)):
    """DivC-PPO (left) and PPO (right) side by side; the shorter clip holds its last frame."""
    crop = crop or SQUARE.get(view)
    a = [frame(left, r, view, size, crop) for r in episode(left, eps[0], last[0])]
    b = [frame(right, r, view, size, crop) for r in episode(right, eps[1], last[1])]
    n = max(len(a), len(b)) + int(hold_s * fps)
    gap = np.full((size[1], GAP, 3), 255, np.uint8)
    write(name, [np.hstack([a[min(i, len(a) - 1)], gap, b[min(i, len(b) - 1)]]) for i in range(n)], fps, poster=-1)


def finger_center(img):
    """Horizontal center of the near finger in the side macro (dark bar just above its tip)."""
    dark = img.mean(2) < 80
    rows = np.where(dark.sum(1) > 0.015 * img.shape[1])[0]
    if not rows.size or rows.max() < 0.2 * img.shape[0]:
        return None
    tip = rows.max()
    band = dark[max(tip - int(0.11 * img.shape[0]), 0):tip - int(0.01 * img.shape[0])]
    cols = np.where(band.sum(0) > 0.5 * band.shape[0])[0]
    return (cols.min() + cols.max()) / 2 if cols.size else None


def macro_frame(run_dir, row, view, size, axis, crop, near_finger=False):
    """Axis-aligned macro of the object top. Dashed line: object axis (always at the image center,
    the camera looks at it). Solid line: gripper axis, placed from the logged error (ee - target).
    Both cameras have image-right = minus the world axis they show, hence the minus sign."""
    img = frame(run_dir, row, view, size, crop)
    h, w = img.shape[:2]
    orange = (img[..., 2] > 180) & (img[..., 1] > 70) & (img[..., 1] < 160) & (img[..., 0] < 80)
    cols = np.where(orange[int(h * 0.92)])[0]
    ppm = (cols.max() - cols.min() + 1) / 15.0 if cols.size > 5 else None
    cx = w // 2
    for y in range(0, h, 12):
        cv2.line(img, (cx, y), (cx, min(y + 6, h - 1)), (60, 60, 60), 1, cv2.LINE_AA)
    e = row["err_mm"][axis]
    color = ((16, 54, 255), (0, 153, 77))[axis]  # BGR of the Fig. 2 X / Y colours
    gx = None
    if near_finger:  # side: the visible finger is closer to the camera than the object axis
        fc = finger_center(img)
        gx = None if fc is None else int(round(fc))
    elif ppm and abs(e) < 40:  # front: both fingers lie in the object plane
        gx = int(round(cx - e * ppm))
    if gx is not None:
        cv2.line(img, (gx, 0), (gx, h - 1), color, 2, cv2.LINE_AA)
    label = f"{'XY'[axis]} error {e:+.2f} mm" if abs(e) < 100 else f"{'XY'[axis]} error {e / 10:+.1f} cm"
    cv2.putText(img, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(img, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return img


def macro_pair(name, left, right, view, size, fps, axis, ep, crop=None, near_finger=False, hold_s=1.2):
    a = [macro_frame(left, r, view, size, axis, crop, near_finger) for r in episode(left, ep)]
    b = [macro_frame(right, r, view, size, axis, crop, near_finger) for r in episode(right, ep)]
    n = max(len(a), len(b)) + int(hold_s * fps)
    gap = np.full((size[1], GAP, 3), 255, np.uint8)
    write(name, [np.hstack([a[min(i, len(a) - 1)], gap, b[min(i, len(b) - 1)]]) for i in range(n)], fps, poster=-1)


def hero_data():
    """Per-axis error (mm) per captured step for the teaser traces (the PPO p85 placement)."""
    hero = {"fps": 10, "policy_hz": POLICY_HZ, "stride": 3}
    for key in ("divc", "ppo"):
        rows = episode(f"{key}_p85_v2_s42", 0)
        hero[key] = {"t_end": rows[-1]["t"], "frames": len(rows), "t": [r["t"] for r in rows],
                     "stable": rows[-1]["label"] == "positioned",
                     "err": [[round(abs(v), 3) for v in r["err_mm"]] for r in rows]}
    (ROOT / "static/hero.json").write_text(json.dumps(hero, separators=(",", ":")))


# one placement per condition, replayed for both policies (DualHeadPPO/.cache/run_page_v2_capture.sh)
CONDITIONS = {"p85": 10, "ood14": 20, "noise": 20, "scale": 20}  # fps: 10 is real time, 20 is 2x


def videos():
    # teaser: fixed camera az60, one frame per 3 policy steps -> 10 fps is real time
    teaser_crop = ((40, 1080), (700, 1856))
    # PPO episode at its 85th-percentile final error (1.9 mm, never stable), same placement replayed for DivC-PPO
    clips = {k: [frame(f"{k}_p85_v2_s42", r, "fixed_az60", (720, 648), teaser_crop) for r in episode(f"{k}_p85_v2_s42", 0)]
             for k in ("divc", "ppo")}
    n = max(map(len, clips.values())) + 12
    for key, f in clips.items():
        write(f"seq_{key}", f + [f[-1]] * (n - len(f)), 10)

    # front macro looks along -Y and shows the X error; side macro looks along +X and shows Y
    MACROS = (("world_macro_front", 0, (768, 432), None, None), ("world_macro_side", 1, (432, 432), SQUARE_MACRO, True))
    for cond, fps in CONDITIONS.items():
        left, right = f"divc_{cond}_v2_s42", f"ppo_{cond}_v2_s42"
        pair(f"{cond}_world_45", left, right, "world_45", (640, 360), fps)
        for view, axis, size, crop, g in MACROS:
            macro_pair(f"{cond}_{view}", left, right, view, size, fps, axis, 0, crop, g)
    pair("p85_wrist_bbox", "divc_p85_v2_s42", "ppo_p85_v2_s42", "wrist_bbox", (450, 450), 10)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    for old in list(IMG.glob("*")) + list(VID.glob("*")):
        old.unlink()
    figures()
    hero_data()
    videos()
    print("done")
