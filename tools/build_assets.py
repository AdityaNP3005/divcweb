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


def single(name, run_dir, ep, views, size, fps, hold_s=1.0):
    rows = episode(run_dir, ep)
    gap = np.full((size[1], GAP, 3), 255, np.uint8)
    frames = []
    for r in rows:
        tiles = [frame(run_dir, r, v, size, SQUARE.get(v)) for v in views]
        frames.append(np.hstack(sum([[t, gap] for t in tiles], [])[:-1]))
    frames += [frames[-1]] * int(hold_s * fps)
    write(name, frames, fps, poster=-1)


def hero_data():
    """Per-axis error (mm) per captured step for the teaser traces."""
    hero = {"fps": 10, "policy_hz": POLICY_HZ, "stride": 3}
    for key, run_dir, t_end in (("divc", "divc_nominal_seq_s42", 47), ("ppo", "ppo_nominal_seq_s42", 65)):
        rows = episode(run_dir, 0, t_end)
        hero[key] = {"t_end": t_end, "frames": len(rows), "t": [r["t"] for r in rows],
                     "err": [[round(abs(v), 3) for v in r["err_mm"]] for r in rows]}
    (ROOT / "static/hero.json").write_text(json.dumps(hero, separators=(",", ":")))


def videos():
    W16 = (800, 450)
    # teaser: fixed camera az60, one frame per 3 policy steps -> 10 fps is real time
    teaser_crop = ((40, 1080), (700, 1856))
    for key, run_dir, t_end in (("divc", "divc_nominal_seq_s42", 47), ("ppo", "ppo_nominal_seq_s42", 65)):
        rows = episode(run_dir, 0, t_end)
        f = [frame(run_dir, r, "fixed_az60", (720, 648), teaser_crop) for r in rows]
        write(f"seq_{key}", f + [f[-1]] * (65 // 3 + 1 - len(f) + 12), 10)

    # nominal, same placement (ep0), stopped at the first stable step
    for view, size in (("overview_az60", W16), ("closeup_side", (540, 540)), ("wrist_bbox", (450, 450))):
        pair(f"nom_{view}", "divc_nominal_seq_s42", "ppo_nominal_seq_s42", view, size, 10, last=(47, 65))

    # PPO near-target failure (ep1): never stable within 7 s
    single("ppo_fail", "ppo_nominal_seq_s42", 1, ("closeup_side", "wrist_bbox"), (450, 450), 20)

    # 14 cm placement radius, same placement per episode, one frame per 10 steps -> 6 fps is 2x speed.
    # The overview camera shows how far the arm travels; front and side cameras exist only at the
    # start and final steps, so they are shown as stills.
    for ep in (0, 1, 2):
        pair(f"ood_ep{ep}", "divc_ood14_s42", "ppo_ood14_s42", "overview_az60", W16, 6, eps=(ep, ep))
        for view in ("overview_front", "overview_side"):
            for tag, run_dir, row in (("start", "divc_ood14_s42", 0), ("divc", "divc_ood14_s42", -1), ("ppo", "ppo_ood14_s42", -1)):
                r = episode(run_dir, ep)[row]
                r = dict(r, label=r["label"]) if r["label"] in ("start", "final") else None
                assert r, (run_dir, ep, row)
                cv2.imwrite(str(IMG / f"ood_ep{ep}_{view.split('_')[1]}_{tag}.webp"),
                            frame(run_dir, r, view, (640, 360)), [cv2.IMWRITE_WEBP_QUALITY, 82])


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    for old in list(IMG.glob("*")) + list(VID.glob("*")):
        old.unlink()
    figures()
    hero_data()
    videos()
    print("done")
