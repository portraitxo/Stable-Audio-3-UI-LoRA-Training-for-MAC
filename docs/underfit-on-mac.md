# Underfit on a Mac — the steps its README leaves out

[Underfit](https://github.com/dada-bots/underfit) is the full LoRA-training
dashboard, and on Apple Silicon it trains by shelling out to the **same**
Stable Audio 3 MLX trainer this repo's UI drives. Its Apple-Silicon quickstart
is four lines long and skips several things a fresh Mac needs, which is why it
fails partway through for most people on the first try.

These notes were checked against `dada-bots/underfit` and
`Stability-AI/stable-audio-3` as of **October 2026**. File and line references
are to those repos, so you can verify any claim here yourself.

> **Short version:** `UNDERFIT_ENGINE=mlx ./run.sh` does not put the dashboard
> on MLX — the dashboard never reads that variable. You pick the engine in the
> **New Finetune** form. See [the engine section](#the-engine-variable-doesnt-do-what-it-looks-like).

---

## Full install, from a laptop with nothing on it

```bash
# 0. Apple's Command Line Tools — this is where git comes from.
#    Skip if `git --version` already prints a version.
xcode-select --install

# 1. ffmpeg. Optional for Stable Audio 3, but you want it for Underfit:
#    without it the dashboard skips its ground-truth audio previews and clip
#    downloads, and training demos save as big WAVs instead of MP3s.
#    (No Homebrew? Install it first: https://brew.sh)
brew install ffmpeg

# 2. Clone BOTH repos into the SAME parent folder, as siblings.
#    The folder MUST be named stable-audio-3 — see "Siblings" below.
cd ~
git clone https://github.com/dada-bots/underfit
git clone https://github.com/Stability-AI/stable-audio-3

# 3. Set up the MLX runtime (its own venv + the MLX weight packs).
#    Answer the bundle picker: sm-music is the fast model, medium is better.
cd ~/stable-audio-3/optimized/mlx && ./install.sh

# 4. Install the dashboard's dependencies.
#    --no-setup is important: it skips the PyTorch-backend wizard, which you
#    don't need on MLX and which is where the HuggingFace 401 errors come from.
cd ~/underfit && ./install.sh --no-setup

# 5. Start the dashboard.
cd ~/underfit && ./run.sh
```

Then open **http://localhost:8787** and, in **+ New Finetune**, check that the
**Engine** dropdown says **MLX** before you start a run.

Step 4 downloads PyTorch (~hundreds of MB) even though MLX training never uses
it — the dashboard imports torch just to draw spectrograms
(`dashboard/server.py:3084`). It looks like the wrong thing is happening. It
isn't. Let it finish.

Underfit is Python **3.10 only** (`pyproject.toml`: `requires-python =
">=3.10,<3.11"`). You don't need to install that yourself — `uv` fetches it. Do
not point it at a system Python or a 3.11+ venv.

---

## The engine variable doesn't do what it looks like

Underfit's README says:

```bash
UNDERFIT_ENGINE=mlx ./run.sh          # dashboard on http://localhost:8787
```

`UNDERFIT_ENGINE` is read by the **command-line** trainer (`lora_train.py:138`)
and by `run_gradio.py:45`. It is **not read anywhere in `dashboard/server.py`**.
The dashboard takes the engine from the New Finetune form, per run, defaulting
to `torch` if the form doesn't say otherwise (`dashboard/server.py:4124`).

In practice the front-end pre-selects MLX for you when MLX is available
(`dashboard/index.html:6354`, and `_available_engines` at
`dashboard/server.py:2801` puts `mlx` first on Apple), so the dropdown usually
says MLX already. But that is the front-end doing it, not the variable — so:

- Setting the variable is harmless. Leave it in if you like.
- **Always look at the Engine dropdown before starting a run.** That is the
  only thing that decides which engine trains.
- MLX showing in the dropdown does **not** mean Underfit found your Stable
  Audio 3 checkout. `_available_engines` offers MLX on any Apple-Silicon Mac
  without checking (`dashboard/server.py:2801`), so a broken layout surfaces
  later, as `MLX trainer script not found` when the run starts — see
  [Siblings](#siblings-the-folder-name-and-location-are-strict).
- If there's **no** MLX option at all, the dashboard didn't detect Apple
  Silicon (`_detect_platform` wants Darwin + arm64, `server.py:2718`). That
  usually means the Terminal is running under Rosetta. Open a normal Terminal.

---

## Siblings: the folder name and location are strict

Underfit finds the MLX runtime at `<the folder containing underfit>/stable-audio-3`
(`underfit/backends/mlx_engine.py:39`). So this works:

```
~/underfit/
~/stable-audio-3/
```

and these do **not**:

```
~/underfit/stable-audio-3/        ← nested inside underfit, not beside it
~/code/underfit/  +  ~/stable-audio-3/    ← different parent folders
~/sa3/                            ← right place, wrong name
```

This is an easy trap, because Stable Audio 3's one-line `bootstrap.sh`
installer clones into *whatever folder you were in when you ran it*. Run it
from inside `underfit/` and you get the nested layout above.

Whenever the layout isn't exactly sibling, name the path yourself:

```bash
UNDERFIT_MLX_ROOT=~/stable-audio-3/optimized/mlx ./run.sh
```

You can also set `UNDERFIT_MLX_PYTHON` if the MLX venv's interpreter isn't at
`<root>/.venv/bin/python`. Unlike `UNDERFIT_ENGINE`, these two **are** read on
the dashboard path (`mlx_engine.py:57-62`).

---

## Why the HuggingFace 401 happens — and why MLX avoids it

Underfit's README warns that the install fails with `401 Unauthorized` unless
you accept the license at
https://huggingface.co/stabilityai/stable-audio-3-medium while signed in to
HuggingFace. That's accurate for the **PyTorch** backend: that repo is
license-gated.

The **MLX** weights are a different repo — `stabilityai/stable-audio-3-optimized`
— and it is **not gated**. Every file the MLX path needs lives there, including
the `-base` checkpoints LoRA training trains against
(`dit_sm-music-base_f16.npz`, `dit_medium-base_f16.npz`, …).

So on the MLX route you need no HuggingFace account, no login, and no license
click. If you hit a 401, you ran the full `./install.sh` wizard instead of
`./install.sh --no-setup`, and it's trying to fetch the gated PyTorch weights.
Either accept the license on HuggingFace, or re-run with `--no-setup`.

---

## Ports

| What | Port | Notes |
| --- | --- | --- |
| Underfit dashboard | 8787 | Taken? It moves to 8788, 8789… and prints which (`server.py:8108`). Force one with `UNDERFIT_DASHBOARD_PORT=9000`. |
| Underfit's per-checkpoint LAUNCH button | 7860+ | `GRADIO_PORT_BASE` (`server.py:505`). |
| This repo's UI, and SA3's own `./sa3-gradio` | 7860+ | Same range. |

So Underfit's LAUNCH button and this repo's UI compete for 7860. Running both
at once isn't harmful — Gradio just walks up to the next free port — but the URL
you expect may belong to the other app. Read the port that's actually printed.

---

## Error-message index

| What you see | What it means |
| --- | --- |
| `no .venv found — run ./install.sh first` | `run.sh` found no venv — step 4 never completed. Scroll back for its real error. |
| `uv: command not found` | `uv` installed into `~/.local/bin` but your current Terminal doesn't have it on PATH. Open a new Terminal window and re-run. |
| `MLX trainer script not found at …` | The sibling layout isn't right. See [Siblings](#siblings-the-folder-name-and-location-are-strict). |
| `MLX venv python not found at …` | Stable Audio 3 is in the right place but its `./install.sh` hasn't run (step 3), or its venv lives elsewhere — then set `UNDERFIT_MLX_PYTHON`. |
| No MLX in the Engine dropdown | The dashboard doesn't see an Apple-Silicon Mac — usually a Terminal running under Rosetta. |
| `401 Unauthorized` / `GatedRepoError` | The gated PyTorch repo. Use `./install.sh --no-setup`, or accept the license on HuggingFace. |
| `NONE_IMPORTABLE` in the dashboard | No PyTorch backend installed. Expected and fine with `--no-setup`, as long as you train on MLX. |
| `git: command not found` | `xcode-select --install`. |
| `Failed to create Metal shared event` | Two things are using the GPU. Train or generate, not both. |
| Dashboard isn't at `localhost:8787` | It took another port. Read the line it printed on startup. |

---

## If the dashboard still won't behave

The MLX trainer underneath it works on its own, and that's the path this repo's
UI uses. Two options, both skipping Underfit entirely:

1. **This repo's UI** — see the [README](../README.md): encode a folder of
   audio, train a LoRA, audition checkpoints, all in a browser tab.
2. **Stable Audio 3's own training CLI** — documented at
   [optimized/mlx → LoRA training](https://github.com/Stability-AI/stable-audio-3/tree/main/optimized/mlx#lora-training).
   Pre-encode, then train:

   ```bash
   cd ~/stable-audio-3/optimized/mlx
   ./.venv/bin/python scripts/pre_encode_mlx.py \
       --audio-dir ~/my-clips --output-dir ~/my-latents --codec same-s
   ./.venv/bin/python scripts/lora_train_mlx.py \
       --dit sm-music --latents-dir ~/my-latents --lr 1e-4 --name my-lora \
       --adapter-type dora-rows --rank 16 --max-steps 2000
   ```

   Use `--codec same-s` with `sm-music`/`sm-sfx` and `same-l` with `medium`.
   Checkpoints land in `output/runs/<name>/<uuid>/checkpoints/`, and this repo's
   UI will load them.

Underfit's dashboard is genuinely nicer when it runs — live loss curves, demo
MP3s with spectrograms, dataset management. These notes are about getting past
the install, not a reason to skip it.
