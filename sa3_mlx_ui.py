#!/usr/bin/env python
"""
sa3_mlx_ui.py — a local Gradio UI for the Stable Audio 3 MLX build.

Two tabs:
  * Generate  — text-to-audio / a2a / inpaint, now with optional LoRA loading.
  * Train LoRA — pre-encode a folder of audio, then train a LoRA on it,
                 with live status and checkpoint auditioning.

Drop this file inside  stable-audio-3/optimized/mlx/  (next to the `sa3`
wrapper and the `scripts/` folder), then:

    cd ~/stable-audio-3/optimized/mlx
    uv pip install gradio          # one-time, into the project's .venv
    ./.venv/bin/python sa3_mlx_ui.py

A local web UI opens in your browser (http://127.0.0.1:7860). Nothing
leaves your machine — it shells out to the scripts in scripts/ using the
same .venv the ./sa3 wrapper uses.

Generation flags (confirmed against sa3_mlx.py):
  --prompt --negative-prompt --dit --decoder --seconds --steps --seed
  --cfg --apg --init-audio --init-noise-level --inpaint-range
  --dit-dtype --free-models/--no-free-models --lora --lora-strength --out
Training flags (confirmed against lora_train_mlx.py --help):
  pre_encode_mlx.py: --audio-dir --output-dir --codec --max-duration
  lora_train_mlx.py: --dit --latents-dir --lr --name --adapter-type
                     --rank --max-steps --checkpoint-every
"""

import datetime
import json
import pathlib
import re
import shlex
import subprocess

import gradio as gr

HERE = pathlib.Path(__file__).resolve().parent
PY = HERE / ".venv" / "bin" / "python"
CLI = HERE / "scripts" / "sa3_mlx.py"
PRE_ENCODE = HERE / "scripts" / "pre_encode_mlx.py"
TRAIN_CLI = HERE / "scripts" / "lora_train_mlx.py"
OUT_DIR = HERE / "output"
RUNS_DIR = OUT_DIR / "runs"
OUT_DIR.mkdir(exist_ok=True)

ANSI = re.compile(r"\x1b\[[0-9;]*m")  # strip terminal colour codes from logs

DIT_CHOICES = ["medium", "sm-music", "sm-sfx"]
DECODER_CHOICES = ["same-l", "same-s"]
# medium pairs with same-l; the small DiTs pair with same-s
SUGGESTED_DECODER = {"medium": "same-l", "sm-music": "same-s", "sm-sfx": "same-s"}
SUGGESTED_CODEC = {"medium": "same-l", "sm-music": "same-s", "sm-sfx": "same-s"}

# One background training process at a time. Stored so Stop/poll can reach it.
TRAIN = {"proc": None, "log": None, "name": None, "fh": None}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _readable_cmd(cmd, wrapper="./sa3"):
    """Human-friendly equivalent of the argv we run (drops python + script)."""
    return " ".join([wrapper] + [shlex.quote(c) for c in cmd[2:]])


def _training_active():
    p = TRAIN["proc"]
    return p is not None and p.poll() is None


def suggest_decoder(dit):
    return gr.update(value=SUGGESTED_DECODER.get(dit, "same-l"))


def suggest_codec(dit):
    return gr.update(value=SUGGESTED_CODEC.get(dit, "same-s"))


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def generate(prompt, negative_prompt, dit, decoder, seconds, steps,
             use_random_seed, seed, cfg, apg,
             init_audio, init_noise_level, inpaint_range,
             dit_dtype, free_models, lora_path, lora_strength):

    if not PY.exists():
        return None, ("ERROR: .venv not found at\n  %s\n\n"
                      "Run ./install.sh inside optimized/mlx first." % PY)
    if not CLI.exists():
        return None, "ERROR: scripts/sa3_mlx.py not found next to this UI."
    if _training_active():
        return None, ("A LoRA training run is active in the Train tab. Running a "
                       "generation now would compete for the GPU (the Metal 'shared "
                       "event' error). Wait for training to finish or Stop it first.")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"ui_{ts}.wav"

    cmd = [
        str(PY), str(CLI),
        "--prompt", prompt or "",
        "--dit", dit,
        "--decoder", decoder,
        "--seconds", str(float(seconds)),
        "--steps", str(int(steps)),
        "--cfg", str(float(cfg)),
        "--apg", str(float(apg)),
        "--dit-dtype", dit_dtype,
        "--out", str(out_path),
    ]
    cmd.append("--free-models" if free_models else "--no-free-models")

    if not use_random_seed and seed is not None:
        cmd += ["--seed", str(int(seed))]
    if negative_prompt and negative_prompt.strip():
        cmd += ["--negative-prompt", negative_prompt]
    if init_audio:  # filepath from the upload component
        cmd += ["--init-audio", init_audio,
                "--init-noise-level", str(float(init_noise_level))]
        if inpaint_range and inpaint_range.strip():
            cmd += ["--inpaint-range", inpaint_range.strip()]
    if lora_path and lora_path.strip():
        cmd += ["--lora", lora_path.strip()]
        if lora_strength is not None and float(lora_strength) != 1.0:
            cmd += ["--lora-strength", str(float(lora_strength))]

    header = "Running:\n  " + _readable_cmd(cmd) + "\n" + ("-" * 60) + "\n"

    try:
        proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    except Exception as e:  # noqa: BLE001
        return None, header + f"Failed to launch the process:\n{e}"

    log = ANSI.sub("", (proc.stdout or "") + "\n" + (proc.stderr or ""))
    if proc.returncode != 0 or not out_path.exists():
        return None, header + f"Generation failed (exit code {proc.returncode}).\n\n{log}"
    return str(out_path), header + log


# --------------------------------------------------------------------------- #
# training: step 1 — pre-encode  (blocking; a few minutes for full tracks)
# --------------------------------------------------------------------------- #
def encode_dataset(audio_dir, latents_dir, codec, trigger):
    if not PY.exists():
        return "ERROR: .venv not found. Run ./install.sh inside optimized/mlx first."
    if not PRE_ENCODE.exists():
        return "ERROR: scripts/pre_encode_mlx.py not found — is your repo up to date? (git pull)"
    if _training_active():
        return ("A training run is active — encoding now would compete for the GPU. "
                "Stop training first.")

    audio_dir = (audio_dir or "").strip()
    latents_dir = (latents_dir or "").strip()
    if not audio_dir:
        return "Enter the path to your audio folder."
    if not latents_dir:
        return "Enter an output (latents) folder."
    src = pathlib.Path(audio_dir).expanduser()
    dst = pathlib.Path(latents_dir).expanduser()
    if not src.is_dir():
        return f"Audio folder not found:\n  {src}"

    cmd = [str(PY), str(PRE_ENCODE),
           "--audio-dir", str(src),
           "--output-dir", str(dst),
           "--codec", codec]
    header = "Encoding:\n  " + _readable_cmd(cmd, wrapper="python scripts/pre_encode_mlx.py") + \
             "\n" + ("-" * 60) + "\n"

    try:
        proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    except Exception as e:  # noqa: BLE001
        return header + f"Failed to launch pre-encode:\n{e}"

    log = ANSI.sub("", (proc.stdout or "") + "\n" + (proc.stderr or ""))
    if proc.returncode != 0:
        return header + f"Encode failed (exit {proc.returncode}).\n\n{log}"

    msg = header + log + "\n" + ("-" * 60) + "\nEncode complete.\n"
    if trigger and trigger.strip():
        msg += "\n" + apply_trigger(str(dst), trigger)
    return msg


def apply_trigger(latents_dir, trigger):
    """Write `trigger` as the 'text' field of every .json sidecar.

    With no --prompt-config, the trainer's build_prompt() falls back to the
    'legacy' path, which returns metadata['text'] when a sidecar carries no
    tag fields (yours don't). So this makes `trigger` the prompt every clip
    trains against — your handle for summoning the style later. An empty
    trigger clears the field (pure baseline).
    """
    d = pathlib.Path(latents_dir).expanduser()
    sidecars = sorted(d.glob("*.json"))
    if not sidecars:
        return f"No .json sidecars in {d} — encode first, then apply the trigger."
    trig = (trigger or "").strip()
    n = 0
    for jf in sidecars:
        try:
            meta = json.loads(jf.read_text())
        except Exception as e:  # noqa: BLE001
            return f"Failed reading {jf.name}: {e}"
        if trig:
            meta["text"] = trig
        else:
            meta.pop("text", None)
        jf.write_text(json.dumps(meta))
        n += 1
    if trig:
        return f"Applied trigger/caption \"{trig}\" to {n} sidecar(s). Train (no prompt-config) will use it."
    return f"Cleared caption on {n} sidecar(s) — training will use empty prompts (baseline)."


# --------------------------------------------------------------------------- #
# training: step 2 — train  (background process + Stop + live poll)
# --------------------------------------------------------------------------- #
def start_training(dit, latents_dir, name, lr, adapter_type, rank,
                   max_steps, checkpoint_every, crop_length):
    if not PY.exists():
        return "ERROR: .venv not found. Run ./install.sh inside optimized/mlx first."
    if not TRAIN_CLI.exists():
        return "ERROR: scripts/lora_train_mlx.py not found — is your repo up to date? (git pull)"
    if _training_active():
        return "A training run is already active. Stop it first, or wait for it to finish."

    latents_dir = (latents_dir or "").strip()
    ld = pathlib.Path(latents_dir).expanduser()
    if not ld.is_dir():
        return f"Latents folder not found:\n  {ld}\nEncode a dataset first."
    if not list(ld.glob("*.npy")):
        return f"No .npy latents in {ld} — did the encode step finish?"

    name = (name or "lora").strip().replace(" ", "-")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logf = OUT_DIR / f"train_{name}_{ts}.log"

    cmd = [str(PY), str(TRAIN_CLI),
           "--dit", dit,
           "--latents-dir", str(ld),
           "--lr", str(lr),
           "--name", name,
           "--adapter-type", adapter_type,
           "--rank", str(int(rank)),
           "--max-steps", str(int(max_steps)),
           "--checkpoint-every", str(int(checkpoint_every))]
    if crop_length:  # 0/blank = use the model default (1300 sm-* / 4096 medium)
        cmd += ["--latent-crop-length", str(int(crop_length))]

    readable = _readable_cmd(cmd, wrapper="python scripts/lora_train_mlx.py")
    try:
        fh = open(logf, "w")
        proc = subprocess.Popen(cmd, cwd=str(HERE),
                                stdout=fh, stderr=subprocess.STDOUT, text=True)
    except Exception as e:  # noqa: BLE001
        return f"Failed to launch training:\n{e}"

    TRAIN.update(proc=proc, log=logf, name=name, fh=fh)
    return ("Training started (running in the background):\n  " + readable +
            f"\n\nLog file: {logf}\n"
            "Status below refreshes every few seconds. Checkpoints appear as they save "
            f"(every {int(checkpoint_every)} steps).")


def stop_training():
    proc = TRAIN["proc"]
    if proc is None:
        return "No training run has been started this session."
    if proc.poll() is not None:
        return f"Training already ended (exit code {proc.returncode})."
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except Exception:  # noqa: BLE001
        proc.kill()
    return ("Stop requested — training was terminated. Any checkpoints already written "
            "(e.g. the last step= milestone) are kept and can still be auditioned.")


def list_checkpoints(name):
    if not name:
        return []
    base = RUNS_DIR / name
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob("*/checkpoints/*.safetensors"))


def poll_training():
    """Called by a Timer + the Refresh button. Cheap: reads a log file."""
    proc = TRAIN["proc"]
    if proc is None:
        return "No training run started yet.", gr.update()

    name = TRAIN["name"]
    logf = TRAIN["log"]
    tail = ""
    if logf and pathlib.Path(logf).exists():
        text = pathlib.Path(logf).read_text(errors="ignore")
        text = ANSI.sub("", text).replace("\r", "\n")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        tail = "\n".join(lines[-25:])

    alive = proc.poll() is None
    status = "RUNNING" if alive else f"FINISHED (exit {proc.returncode})"
    cks = list_checkpoints(name)
    header = f"[{status}]  run: {name}   checkpoints so far: {len(cks)}\n" + ("-" * 60) + "\n"
    latest = cks[-1] if cks else None
    return header + tail, gr.update(choices=cks, value=latest)


def refresh_checkpoints():
    cks = list_checkpoints(TRAIN["name"])
    return gr.update(choices=cks, value=(cks[-1] if cks else None))


def use_checkpoint(ck):
    """Send a chosen checkpoint over to the Generate tab's LoRA field."""
    return gr.update(value=ck or "")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Stable Audio 3 — MLX (local)") as demo:
    gr.Markdown(
        "# Stable Audio 3 · MLX (local)\n"
        "Runs entirely on your Mac via the local `.venv`. "
        "First run of each model downloads its weights from Hugging Face.\n"
        "**Tip:** don't run generation and training at the same time — they'd "
        "compete for the GPU."
    )

    with gr.Tabs():
        # ----------------------------------------------------------------- #
        # GENERATE
        # ----------------------------------------------------------------- #
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(label="Prompt", lines=2,
                                        placeholder="e.g. A beautiful piano arpeggio grows into a cinematic climax 120 BPM")
                    with gr.Row():
                        dit = gr.Dropdown(DIT_CHOICES, value="medium", label="DiT model (--dit)")
                        decoder = gr.Dropdown(DECODER_CHOICES, value="same-l", label="Decoder (--decoder)")
                    with gr.Row():
                        seconds = gr.Slider(1, 380, value=30, step=1, label="Seconds (--seconds)")
                        steps = gr.Slider(1, 8, value=8, step=1, label="Steps (--steps)")

                    with gr.Accordion("LoRA (optional)", open=False):
                        gr.Markdown("Load a trained adapter. The **Train LoRA** tab can drop a "
                                    "checkpoint path in here for you.")
                        lora_path = gr.Textbox(label="LoRA checkpoint (--lora)",
                                               placeholder="output/runs/<name>/<uuid>/checkpoints/<name>-step=….safetensors")
                        lora_strength = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                                                  label="LoRA strength (--lora-strength) · lower if it sounds too much like the data")

                    with gr.Accordion("Guidance (CFG)", open=False):
                        gr.Markdown("`--cfg 1.0` = off. Negative prompt and APG only apply when CFG ≠ 1.0.")
                        cfg = gr.Slider(0.0, 8.0, value=1.0, step=0.1, label="CFG scale (--cfg)")
                        apg = gr.Slider(0.0, 4.0, value=1.0, step=0.1, label="APG (--apg)")
                        negative_prompt = gr.Textbox(label="Negative prompt (--negative-prompt)",
                                                    placeholder="e.g. drums, vocals")

                    with gr.Accordion("Audio-to-audio / Inpainting", open=False):
                        gr.Markdown("Upload a **44.1 kHz, 16-bit PCM WAV** for variation or inpainting.")
                        init_audio = gr.Audio(label="Init audio (--init-audio)", type="filepath")
                        init_noise_level = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                                                    label="Init noise level σ (--init-noise-level) · 0.4–0.8 = variation, 1.0 = full regen")
                        inpaint_range = gr.Textbox(label="Inpaint range (--inpaint-range)",
                                                  placeholder='e.g. 4,7  (regenerate seconds 4–7, keep the rest)')

                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            use_random_seed = gr.Checkbox(value=True, label="Random seed")
                            seed = gr.Number(value=0, precision=0, label="Seed (--seed, used only if Random is off)")
                        dit_dtype = gr.Radio(["fp16", "fp32"], value="fp16", label="DiT dtype (--dit-dtype)")
                        free_models = gr.Checkbox(value=True, label="Free models progressively (--free-models)")

                    go = gr.Button("Generate", variant="primary")

                with gr.Column(scale=2):
                    audio_out = gr.Audio(label="Output", type="filepath")
                    log_out = gr.Textbox(label="Log (command, seed, saved path)", lines=18)

            dit.change(suggest_decoder, inputs=dit, outputs=decoder)
            go.click(
                generate,
                inputs=[prompt, negative_prompt, dit, decoder, seconds, steps,
                        use_random_seed, seed, cfg, apg,
                        init_audio, init_noise_level, inpaint_range,
                        dit_dtype, free_models, lora_path, lora_strength],
                outputs=[audio_out, log_out],
            )

        # ----------------------------------------------------------------- #
        # TRAIN LORA
        # ----------------------------------------------------------------- #
        with gr.Tab("Train LoRA"):
            gr.Markdown(
                "Two steps: **Encode** a folder of audio into latents once, then "
                "**Train** against those latents as many times as you like. "
                "Keeping them separate means experiments (steps, rank, trigger) "
                "don't re-encode your audio every time."
            )

            with gr.Row():
                # ---- Step 1: encode ----
                with gr.Column():
                    gr.Markdown("### 1 · Encode dataset")
                    t_audio_dir = gr.Textbox(
                        label="Audio folder",
                        placeholder="/Users/you/pxo-train-clips  (a folder of audio files)")
                    t_latents_dir = gr.Textbox(
                        label="Latents output folder",
                        value=str(pathlib.Path("~/my-latents").expanduser()))
                    t_codec = gr.Dropdown(DECODER_CHOICES, value="same-s",
                                          label="Codec (--codec) · same-s for sm-music/sm-sfx, same-l for medium")
                    t_trigger = gr.Textbox(
                        label="Trigger word / caption (optional)",
                        placeholder="e.g. portrait-xo   — the phrase you'll use later to summon this LoRA")
                    with gr.Row():
                        encode_btn = gr.Button("Encode dataset", variant="primary")
                        apply_trigger_btn = gr.Button("Apply trigger to existing latents")
                    encode_log = gr.Textbox(label="Encode log", lines=14)

                # ---- Step 2: train ----
                with gr.Column():
                    gr.Markdown("### 2 · Train")
                    t_dit = gr.Dropdown(["sm-music", "sm-sfx", "medium"], value="sm-music",
                                        label="Model (--dit) · sm-music = fast first run")
                    t_train_latents = gr.Textbox(
                        label="Latents folder (--latents-dir)",
                        value=str(pathlib.Path("~/my-latents").expanduser()))
                    t_name = gr.Textbox(label="Run name (--name)", value="album")
                    with gr.Row():
                        t_rank = gr.Slider(2, 64, value=16, step=1, label="Rank (--rank)")
                        t_lr = gr.Number(value=1e-4, label="Learning rate (--lr)")
                    with gr.Row():
                        t_max_steps = gr.Number(value=10000, precision=0, label="Max steps (--max-steps)")
                        t_ckpt_every = gr.Number(value=1000, precision=0, label="Checkpoint every (--checkpoint-every)")
                    t_adapter = gr.Dropdown(["dora-rows", "lora", "dora", "lora-xs"],
                                            value="dora-rows", label="Adapter type (--adapter-type)")
                    t_crop = gr.Number(
                        value=256, precision=0,
                        label="Crop length (--latent-crop-length) · lower = less GPU memory. "
                              "medium OOMs on full tracks — keep this small (256–512). 0 = model default.")
                    with gr.Row():
                        start_btn = gr.Button("Start training", variant="primary")
                        stop_btn = gr.Button("Stop training", variant="stop")
                    train_status = gr.Textbox(label="Training status (auto-refreshes)", lines=14)
                    with gr.Row():
                        ckpt_dropdown = gr.Dropdown([], label="Checkpoints", interactive=True)
                        refresh_btn = gr.Button("Refresh")
                    use_ckpt_btn = gr.Button("Use selected checkpoint in Generate tab →")

            gr.Markdown(
                "Guidance: ~10k steps is a reasonable target (per Underfit's own docs — "
                "past ~20k it tends to overfit). The best LoRA is often **not** the final "
                "checkpoint — audition earlier ones and keep the one that has the style but "
                "still varies on new prompts."
            )

            # wiring
            t_dit.change(suggest_codec, inputs=t_dit, outputs=t_codec)
            encode_btn.click(encode_dataset,
                             inputs=[t_audio_dir, t_latents_dir, t_codec, t_trigger],
                             outputs=encode_log)
            apply_trigger_btn.click(apply_trigger,
                                    inputs=[t_latents_dir, t_trigger],
                                    outputs=encode_log)
            start_btn.click(start_training,
                            inputs=[t_dit, t_train_latents, t_name, t_lr, t_adapter,
                                    t_rank, t_max_steps, t_ckpt_every, t_crop],
                            outputs=train_status)
            stop_btn.click(stop_training, outputs=train_status)
            refresh_btn.click(refresh_checkpoints, outputs=ckpt_dropdown)
            use_ckpt_btn.click(use_checkpoint, inputs=ckpt_dropdown, outputs=lora_path)

            # live status: poll every 3s
            timer = gr.Timer(3.0)
            timer.tick(poll_training, outputs=[train_status, ckpt_dropdown])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()
