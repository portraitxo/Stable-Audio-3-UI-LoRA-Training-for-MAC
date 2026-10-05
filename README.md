# sa3-mlx-ui

A small local **Gradio UI for the Stable Audio 3 MLX build** — generate audio
and **train LoRAs on Apple Silicon**, from a browser tab, without the terminal.

> **Unofficial community tool.** This is a thin wrapper I built around
> Stability AI's `stable-audio-3` `optimized/mlx` runtime. It is not affiliated
> with or endorsed by Stability AI. It ships no model code and no weights — it
> just drives the scripts already in your local SA3 install.

![Stable Audio 3 MLX UI — Generate tab](docs/screenshot4.png)

![Stable Audio 3 MLX UI — Train LoRA tab](docs/screenshot.png)

## What's new

**Blend up to 4 LoRAs at once.** The Generate tab's LoRA panel is now four rows
instead of one. Each row takes a checkpoint, its own **strength**, and an
optional **steps** range, so you can stack a texture LoRA over a rhythm LoRA,
dial each one in separately, and let the base model keep control of the parts
you don't want overwritten. The app also checks every checkpoint's recorded base
model before it runs and tells you if one doesn't match the selected DiT —
instead of letting the CLI fail with a shape error partway through.

## What it does

- **Generate tab** — text-to-audio, audio-to-audio, and inpainting, with CFG /
  negative-prompt / APG controls and **multi-LoRA loading** — blend up to **4**
  adapters at once, each with its own strength and step range (see
  [Blending LoRAs](#blending-loras)).
- **Train LoRA tab** — a two-step workflow:
  1. **Encode** a folder of audio into latents (once).
  2. **Train** a LoRA against those latents (as many times as you like), with a
     live status readout, a **Stop** button, an adjustable **crop length** (to
     keep memory in check on `medium`), an optional **trigger word**, and a
     checkpoint list you can send straight to the Generate tab to audition.

Everything runs locally on your Mac via the SA3 `.venv`. Nothing is uploaded.

## Prerequisites — install Stable Audio 3 first

This UI is a wrapper. It ships no model code and no weights: it drives the
scripts inside a working **Stable Audio 3 MLX** install, using the `.venv` that
install creates. So Stable Audio 3 has to be working *before* you copy the UI in.

Work through this whole section first. Every command is meant to be pasted into
**Terminal** exactly as written, one block at a time.

### 1. Check the Mac can run it

```bash
uname -m
```

Must print **`arm64`** (Apple Silicon, M1 or later). If it prints `x86_64` on an
M-series Mac, your Terminal is running under Rosetta — use a normal Terminal
window. On an actual Intel Mac, none of this will work: MLX is Metal-only.

Free disk space you'll need, depending on which model you use:

| What you're doing | Download |
| --- | --- |
| `sm-music` or `sm-sfx`, generating only | ~1.9 GB (1.3 GB model + 0.5 GB text encoder) |
| …and training LoRAs on it | ~2.8 GB (the trainer adds a 0.9 GB base checkpoint) |
| `medium`, generating only | ~6.5 GB |
| …and training LoRAs on it | ~9.2 GB (its base checkpoint is 2.8 GB) |

Start with `sm-music`. It's small, fast, and enough to prove the whole pipeline
works. You can add `medium` later with one command.

**You do not need a HuggingFace account, a login, or a token.** The MLX weights
live in `stabilityai/stable-audio-3-optimized`, which is not license-gated. (If
you see a `401 Unauthorized` or `GatedRepoError` anywhere, you're running some
*other* Stable Audio installer that pulls the gated PyTorch weights — not this
one.)

### 2. Install Apple's Command Line Tools

This is where `git` comes from. Skip if `git --version` already prints a version.

```bash
xcode-select --install
```

A dialog appears — click **Install** and wait for it to finish (a few minutes).
Then check:

```bash
git --version
```

### 3. Download Stable Audio 3

```bash
cd ~
git clone https://github.com/Stability-AI/stable-audio-3
```

That makes `~/stable-audio-3`. Keep that name and location if you can — the
commands below and the launcher's default both assume it.

### 4. Run Stable Audio 3's installer

```bash
cd ~/stable-audio-3/optimized/mlx
./install.sh -y --download sm-music
```

The two flags matter on other people's machines: `-y` answers the "install uv?"
prompt for you, and `--download sm-music` picks the model bundle up front
instead of stopping at an interactive menu. For the bigger model, or both:

```bash
./install.sh -y --download medium
./install.sh -y --download sm-music,medium
```

Re-running is safe — it skips anything already present.

What it does, so you know what you're watching: installs `uv` if missing,
creates `.venv` with Python 3.11 inside `optimized/mlx`, installs the MLX
dependencies into it, notes whether `ffmpeg` is present (optional — it's fine
without), then downloads the weights you asked for from HuggingFace.

### 5. If `./install.sh` doesn't finish

It isn't reliable on a fresh machine. Match the symptom:

| What you see | What to do |
| --- | --- |
| `uv was installed but isn't on PATH` | Quit Terminal, open a new window, and re-run the same command. The installer only adds `uv` to the PATH of the shell it's running in. |
| It stops at a menu asking which models to download | You left off `--download`. Press Ctrl-C and re-run with `-y --download sm-music`. |
| `curl is required` / the `uv` install fails | Install `uv` with Homebrew instead: `brew install uv`, then re-run the command in step 4. No Homebrew? Get it at https://brew.sh first. |
| A download dies partway, or the weights stall | Re-run the step-4 command. Finished files are skipped, so it picks up where it stopped. |
| `warning: this stack is Apple-Silicon-only` | See step 1 — Rosetta, or an Intel Mac. |
| Anything else, or it just won't go | Use the by-hand path below. |

**By-hand path** — this does exactly what `install.sh` does, without relying on
it. It needs Homebrew (https://brew.sh):

```bash
# A Python the MLX stack accepts (3.10 or newer)
brew install python@3.11

# Build the virtual environment yourself
cd ~/stable-audio-3/optimized/mlx
"$(brew --prefix)/bin/python3.11" -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

# Download the weights (bundle names: sm-music, sm-sfx, medium)
./.venv/bin/python scripts/install.py --download sm-music
```

All of those dependencies are prebuilt wheels — nothing compiles, so there's
nothing here that can fail on a missing compiler.

If even the weight download misbehaves, skip it. Stable Audio 3 fetches any
missing weight file from HuggingFace the first time it needs it, so a venv with
the dependencies installed is enough to continue — the first generation will
just take longer while it downloads.

### 6. Prove Stable Audio 3 works before touching the UI

Don't skip this. If this command produces a file, the UI will work; if it
doesn't, the UI can only fail in a more confusing way.

```bash
cd ~/stable-audio-3/optimized/mlx
./.venv/bin/python scripts/sa3_mlx.py \
    --prompt "short warm techno loop" --dit sm-music --decoder same-s \
    --seconds 5 --out check.wav
```

First run is slow — it loads (and possibly downloads) the text encoder and the
model. When it finishes you'll have `~/stable-audio-3/optimized/mlx/output/check.wav`.
Play it:

```bash
afplay ~/stable-audio-3/optimized/mlx/output/check.wav
```

Now go to [Install](#install).

### 7. Extra notes if you plan to train LoRAs

Training is the reason most people are here, so two things to know before you
start:

- **Training downloads one more file.** LoRA training runs against the model's
  **base** checkpoint, not the one generation uses, and it isn't part of any
  install bundle — the trainer fetches it on your first training run (0.9 GB for
  `sm-music`/`sm-sfx`, 2.8 GB for `medium`). Nothing to do in advance; just
  don't be surprised by a download when you first hit Train, and leave the disk
  space for it.
- **Match the codec to the model.** Encode your audio with `same-s` for
  `sm-music`/`sm-sfx`, and `same-l` for `medium`. Latents encoded for one won't
  train the other, so use a separate output folder per model. The UI exposes
  this choice on the Train tab — it's the single most common way a first
  training run goes wrong.

The underlying command-line trainer is documented at
[optimized/mlx → LoRA training](https://github.com/Stability-AI/stable-audio-3/tree/main/optimized/mlx#lora-training).
This UI drives those same scripts, so that page is the reference for what every
option actually does.

## Install

Stable Audio 3 working? Two blocks. Every path is absolute (`~/…`), so it
doesn't matter which folder Terminal is in.

**1 — Download this UI and copy it into your Stable Audio 3 folder:**

```bash
cd ~ && git clone https://github.com/portraitxo/Stable-Audio-3-UI-LoRA-Training-for-MAC sa3-mlx-ui
cp ~/sa3-mlx-ui/sa3_mlx_ui.py ~/sa3-mlx-ui/sa3-ui.command ~/stable-audio-3/optimized/mlx/
chmod +x ~/stable-audio-3/optimized/mlx/sa3-ui.command
```

(The `git clone` is the step people miss. Without it there's no
`sa3_mlx_ui.py` to copy, and `cp` fails with `No such file or directory`.)

**2 — Install Gradio into the Stable Audio 3 environment:**

```bash
~/stable-audio-3/optimized/mlx/.venv/bin/python -m pip install gradio
```

Use the `.venv`'s own `pip`, exactly as written — not `uv pip install`. Stable
Audio 3's installer only puts `uv` on the PATH for the length of its own run, so
in a fresh Terminal window `uv` is usually `command not found`.

(If your `stable-audio-3` lives somewhere other than `~/stable-audio-3`, adjust
these paths and edit `MLX_DIR` at the top of `sa3-ui.command`.)

## Run

```bash
cd ~/stable-audio-3/optimized/mlx && ./.venv/bin/python sa3_mlx_ui.py
```

Then open http://127.0.0.1:7860. Or just double-click `sa3-ui.command` in
Finder — it does the same thing and opens the browser for you.

Port 7860 is the Gradio default, so other audio UIs tend to want it too. If it's
already taken, Gradio moves to the next free port (7861, 7862…) — read the URL
printed in the Terminal rather than assuming 7860.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `cp: sa3_mlx_ui.py: No such file or directory` | The `git clone` in **Install** step 1 was skipped, so there's nothing to copy. Run that whole block. |
| `git: command not found`, or a popup about developer tools | `xcode-select --install`, wait for it to finish — **Prerequisites** step 2. |
| `uv: command not found` | Normal in a new Terminal window — the SA3 installer only puts `uv` on its own PATH. Use the `.venv` pip from **Install** step 2, or `brew install uv`. |
| `No virtual environment at …/.venv/bin/python` | Stable Audio 3 isn't installed yet — **Prerequisites** steps 3–4 (and step 5 if its installer won't finish). |
| `ModuleNotFoundError: No module named 'gradio'` | **Install** step 2 didn't run, or it went into a different Python. Re-run it with the full `~/stable-audio-3/optimized/mlx/.venv/bin/python` path. |
| `permission denied: ./sa3-ui.command`, or double-clicking it does nothing | `chmod +x ~/stable-audio-3/optimized/mlx/sa3-ui.command` |
| `ModuleNotFoundError: No module named 'mlx'`, or the UI starts but every run errors | Stable Audio 3's own install is incomplete. Go back to **Prerequisites** step 6 and get that generating a file first. |
| `Could not find: …/stable-audio-3/optimized/mlx` (from the launcher) | Your SA3 install is somewhere else — edit `MLX_DIR` at the top of `sa3-ui.command`. |
| `address already in use` | Another app has port 7860. Quit it, or use whichever 786x URL the Terminal prints. |
| `Failed to create Metal shared event` | You're generating and training at the same time. Run one at a time. |
| Model weights download every time / fills the disk | Weights land in the HuggingFace cache and are symlinked into `models/mlx/`. Don't delete that cache between runs. |

## Using Underfit on a Mac

[Underfit](https://github.com/dada-bots/underfit) is the full LoRA-training
dashboard, and on Apple Silicon it drives the **same** MLX trainer this UI does.
Its Apple-Silicon quickstart leaves out a few things a fresh Mac needs, and
`UNDERFIT_ENGINE=mlx ./run.sh` doesn't actually put the dashboard on MLX — the
dashboard never reads that variable.

**[→ Underfit on a Mac: the steps its README leaves out](docs/underfit-on-mac.md)**
— a full fresh-laptop install, the strict sibling-folder rule, why the
HuggingFace 401 happens (and why the MLX route avoids it), the ports, and an
error-message index.

## Blending LoRAs

The **LoRA — blend up to 4** accordion on the Generate tab has four identical
rows. Fill in as many as you want and leave the rest empty:

| Field | What it does |
| --- | --- |
| **checkpoint** | Path to a `.safetensors` LoRA. The **Train LoRA** tab's "use checkpoint" button drops one into the first empty row. |
| **strength** | How hard that adapter pulls (0–2). Lower it when a LoRA is overpowering the others or sounds too much like its training data. |
| **steps** | Which denoising steps the LoRA applies on — `2-8`, `2-`, `-4`, or a single `3`. Empty means all steps. |

The **Default strength** slider above the rows sets `--lora-strength` for the
run; per-row strengths override it.

The **steps** range is the useful lever when you stack adapters. Skipping the
early steps (e.g. `3-`) lets the base model lay down structure and form before
the LoRA colors it in — often the difference between a blend that sounds musical
and one that collapses into training-data mush. Conversely, `-4` applies a LoRA
only while the broad shape is being decided and leaves the detail alone.

**All loaded LoRAs must be trained on the same base model as the one you're
generating with** — you can't mix a `medium` LoRA with a small model, or vice
versa. The UI reads each checkpoint's metadata and stops with an explanation
before launching, so a mismatch costs you a message instead of a crash.

## Notes & gotchas

- **Don't generate and train at the same time** — both use the GPU and will
  compete for Metal resources (you may see a "Failed to create Metal shared
  event" error). Run one at a time, and quit other GPU-heavy apps while
  training.
- **Match the model to its codec/latents.** `sm-music`/`sm-sfx` pair with the
  `same-s` codec; `medium` pairs with `same-l`. Latents encoded for one won't
  train the other. Encode into a separate folder per model.
- **LoRAs can't cross model families.** Every LoRA loaded in one generation
  must match the selected DiT's base (all `medium`, or all small). The UI checks
  this for you before running.
- **`medium` is memory-hungry.** Training on full-length latents can exhaust GPU
  memory even on large Macs. The **crop length** control defaults to a small
  value for this reason; raise it only if training is stable.
- **Trigger word.** With no `--prompt-config`, the SA3 trainer falls back to a
  legacy prompt path that reads a `text` field from each latent's `.json`
  sidecar. The "trigger" field writes your phrase there so every clip trains
  against it — your handle for summoning the style at generation time. This
  depends on the trainer's current behavior; verify by training a short run and
  checking the trigger word actually steers generation.
- **Steps.** ~10k is a common target for a LoRA; the best checkpoint is often
  not the final one, so audition earlier checkpoints too.

## Requirements

- macOS on Apple Silicon (M1 or later)
- A working `stable-audio-3` `optimized/mlx` install, with its `.venv` — see
  [Prerequisites](#prerequisites--install-stable-audio-3-first)
- ~2 GB of free disk for the smallest model, ~9 GB for `medium` with training
- `gradio` (see `requirements.txt`) — the only thing this UI adds
- No HuggingFace account, and no PyTorch

## Licensing & weights

- **This UI code** is MIT-licensed (see `LICENSE`).
- **Stable Audio 3, its `optimized/mlx` runtime, and its model weights are
  NOT covered by that license.** They belong to Stability AI and are governed
  by the **Stability AI Community License** and the model's own terms, which
  include a commercial-use revenue threshold. This repo redistributes none of
  those weights or that code — you obtain them yourself through the official
  SA3 install, and by doing so you accept Stability's license. Review Stability
  AI's current terms before any commercial use; I'm not a lawyer and nothing
  here is legal advice.

## Acknowledgments

This UI stands entirely on other people's work:

- **Stability AI** — Stable Audio 3 and the `optimized/mlx` runtime this UI
  wraps.
- **Dada Bots — Underfit** — the LoRA training conventions the MLX trainer
  mirrors.
- **@betweentwomidnights** — the MLX LoRA-training path for Apple Silicon.

If you build on this, please keep these credits.

— Portrait XO
