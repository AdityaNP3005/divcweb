# DivC-PPO project page

Project page for the paper

> **DivC-PPO: Cartesian-Aligned Credit Assignment for Precision Vision-Guided End-Effector Positioning**
> Aditya Nugraha Putra, Van-Linh Nguyen. National Chung Cheng University.

The page is static HTML and CSS. It has no build step and no JavaScript framework.

## Contents

| Section | Source |
|---|---|
| Teaser: synchronized DivC-PPO and PPO episode with per-axis error traces | Isaac Sim rollout, seed 42 |
| Abstract | Paper abstract |
| Method | Paper Fig. 2 and its caption |
| Results table and error distribution | Paper Table III and Fig. 3 |
| Front and side views of the teaser episode | Isaac Sim rollout, seed 42 |
| Deployment perturbations (14 cm placement, detection noise, action scale) | Paper Table V and Isaac Sim rollouts, seed 42 |

The text and three-seed numbers come from the paper. The per-episode numbers come from the logs of the rendered rollouts. In every comparison video, both policies start from the same object placement.

## Repository layout

```
index.html              page markup and the small inline script (video sync, traces, tabs)
style.css               overrides on Bulma 0.9.4
static/hero.json        per-axis error per frame for the teaser traces
static/img/             figures and video posters (WebP)
static/video/           comparison videos (VP9 WebM)
tools/build_assets.py   regenerates everything under static/
```

External assets are loaded from public CDNs:
- Bulma 0.9.4 and Font Awesome 6.4.2 from cdnjs.
- Google Sans and Noto Sans from Google Fonts.

## Local preview

```
python3 -m http.server 8765
```

Then open <http://localhost:8765>.

## Rebuilding the assets

```
python3 tools/build_assets.py
```

The script needs `pdftocairo`, ImageMagick and OpenCV with a VP9 WebM writer. It reads the paper figures (PDF) and the frames rendered from the trained policies. Both are kept in the authors' research repositories and are not included here. The paths are set at the top of the script. Frame rates and playback speeds are noted in the script and on the page.

## Deployment

The site is deployed on Vercel from this repository with these settings:

| Setting | Value |
|---|---|
| Framework preset | Other |
| Build command | none |
| Output directory | `.` |

## After publication

- Paper: replace the disabled `Paper (to appear)` button in `index.html` with a link to the paper.
- Code: replace the disabled `Code (coming soon)` button with a link once the code is public.
- BibTeX: add the venue, pages and DOI.

## Acknowledgment

The page layout follows the [Nerfies](https://github.com/nerfies/nerfies.github.io) project page template.
