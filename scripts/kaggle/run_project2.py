"""Kaggle run for Project 2 (abstention training vs runtime gating). See docs/PREREG.md.

Bundle (private Kaggle dataset): gatedrag/ (code, configs, chunks), abstain-or-gate/ (this repo),
and a frozen golden split: <golden>/dev.jsonl, <golden>/test.jsonl, <golden>/MANIFEST.json.

Phases (each output is saved as soon as it exists; a rerun skips finished steps):
  1  base model open-loop pool on DEV (K = 4)            -> SFT data + DPO pairs, calibrator for arm 5
  2  base arms on TEST; lm-eval side effects for the base model
  3  per seed (0, 1, 2): QLoRA SFT -> merge -> DPO -> merge, then its trained arms on TEST through
     gatedrag (K = 2), lm-eval side effects (seed 0 only), then its merged models are deleted
  5  contamination report, analysis -> REPORT.md / report.json

GATEDRAG_DRYRUN=1: fake generator for pools, a tiny model for training, no lm-eval. It checks the
orchestration end to end on a laptop; nothing it produces is a result.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

DRY = os.environ.get("GATEDRAG_DRYRUN") == "1"
BASE = os.environ.get("P2_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
SEEDS = (0, 1) if DRY else (0, 1, 2)
K_DEV, K_TEST = (3, 2) if DRY else (8, 2)   # 8 samples per dev item: more distinct answers to pair
K_SC = 3 if DRY else 5                        # samples for the self-consistency baseline arm
JUDGE = "mistralai/Mistral-7B-Instruct-v0.3"
INPUT = Path(os.environ.get("P2_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("P2_OUT", "/kaggle/working/out"))
PORT = 8012
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(msg: str) -> None:
    line = f"[{(time.time() - T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


def find(pattern: str) -> Path:
    hits = glob.glob(str(INPUT / "**" / pattern), recursive=True)
    if not hits:
        raise SystemExit(f"not found in input: {pattern}")
    return Path(hits[0])


gd_root = Path(os.environ["P2_GATEDRAG"]) if "P2_GATEDRAG" in os.environ else find("gatedrag/__init__.py").parent.parent.parent
ag_root = Path(os.environ["P2_SELF"]) if "P2_SELF" in os.environ else find("abstaingate/__init__.py").parent.parent.parent
golden = Path(os.environ["P2_GOLDEN"]) if "P2_GOLDEN" in os.environ else find("MANIFEST.json").parent
chunks = Path(os.environ.get("P2_CHUNKS", gd_root / "data/processed/chunks.jsonl"))
sys.path[:0] = [str(gd_root / "src"), str(ag_root / "src")]
# Three environments, because their pins conflict (vLLM 0.6.6 pins torch 2.5.1 and an older
# transformers than the training stack): training in the kernel's own Python, pinned to exactly the
# versions the local dry run used; vLLM and lm-eval each in an isolated venv, called as processes.
TRAIN_PINS = ["trl==1.18.1", "peft==0.21.3", "transformers==5.23.0", "datasets", "accelerate", "bitsandbytes",
              "opentelemetry-api", "opentelemetry-sdk", "httpx"]


def venv(name: str, *pkgs: str) -> str:
    path = Path(f"/tmp/venv_{name}")              # not /kaggle/working: that is size-limited run output
    if not (path / "bin/python").exists():
        # Kaggle's Python (3.13) cannot build a stdlib venv, and vLLM 0.6.6 needs Python <= 3.12:
        # uv fetches a standalone 3.12 and installs into it
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        subprocess.run([sys.executable, "-m", "uv", "venv", "-q", "--python", "3.12", str(path)], check=True)
        subprocess.run([sys.executable, "-m", "uv", "pip", "install", "-q", "--python", str(path / "bin/python"), *pkgs],
                       check=True)
    return str(path / "bin/python")


if not DRY:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *TRAIN_PINS], check=True)
    # Kaggle preinstalls torchao 0.10; peft 0.21 refuses any torchao below 0.16 when it merges adapters.
    # Nothing here uses torchao, so remove it rather than upgrade it.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    VLLM_PY = venv("vllm", "vllm==0.6.6.post1", "transformers==4.47.1")
    LMEVAL_PY = venv("lmeval", "lm-eval[hf]==0.4.9", "transformers==4.46.3")

from gatedrag.cli import make_generator  # noqa: E402
from gatedrag.core.config import load_config  # noqa: E402
from gatedrag.data.corpus import load_chunks  # noqa: E402
from gatedrag.evaluation.calibration import answered_pairs, fit_isotonic  # noqa: E402
from gatedrag.evaluation.golden import load_golden  # noqa: E402
from gatedrag.evaluation.runner import load_pool, run_pool, save_pool  # noqa: E402
from gatedrag.models.pipeline import Pipeline  # noqa: E402
from gatedrag.models.retrieval import Retriever  # noqa: E402

from abstaingate.data.pairs import build, write_jsonl  # noqa: E402
from abstaingate.data.prompts import prompt_builder  # noqa: E402
from abstaingate.evaluation import contamination  # noqa: E402
from abstaingate.evaluation.analysis import analyse, render  # noqa: E402
from abstaingate.models.train import TrainConfig  # noqa: E402

dev_items, test_items = load_golden(golden / "dev.jsonl"), load_golden(golden / "test.jsonl")
dev_raw = [json.loads(x) for x in (golden / "dev.jsonl").read_text().splitlines() if x.strip()]
test_ids = {g.id for g in test_items}
cfg0 = load_config(gd_root / "configs/release.toml")
retriever = Retriever(load_chunks(chunks))
manifest = json.loads((golden / "MANIFEST.json").read_text())
log(f"golden {golden.name}: dev {len(dev_items)}, test {len(test_items)}, pilot_only={manifest.get('pilot_only')}")


def serve(model: str, tp: int = 1):
    """vLLM OpenAI server for `model`; returns (process, generator). Dry run: the fake generator."""
    if DRY:
        return None, make_generator("fake")
    cmd = [VLLM_PY, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", str(PORT),
           "--dtype", "half", "--max-model-len", "4096", "--gpu-memory-utilization", "0.85", "--seed", "0",
           "--tensor-parallel-size", str(tp)]
    srv_log = open(OUT / f"vllm_{Path(model).name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited for {model}; see {srv_log.name}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
            return proc, make_generator(f"openai:http://127.0.0.1:{PORT}|{model}")
        except Exception:
            time.sleep(10)
    proc.terminate()
    raise RuntimeError(f"vLLM did not come up for {model}")


def stop(proc) -> None:
    """Kill vLLM's whole process group (its workers hold GPU memory), then wait until the GPU is free;
    training starts right after and needs that memory."""
    if proc is None:
        return
    import signal
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(60)
            break
        except subprocess.TimeoutExpired:
            continue
    for _ in range(60):
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        used = [int(x) for x in r.stdout.split()] if r.returncode == 0 else []
        if not used or max(used) < 1500:
            return
        time.sleep(5)
    log("warning: GPU memory still in use after stopping vLLM")


def pool(name: str, gen, cfg, items, k: int, calibrator=None) -> dict:
    path = OUT / f"{name}.json"
    if path.exists():
        return load_pool(path)
    p = run_pool(Pipeline(retriever, gen, cfg, calibrator=calibrator), items, k=k, concurrency=1 if DRY else 16)
    save_pool(p, path)
    log(f"{name}: {k} runs x {len(items)} items")
    return p


def calibrator_from(dev_pool: dict):
    s, c = answered_pairs(dev_pool)
    return fit_isotonic(s, c) if s.size >= 20 else None


open_loop = cfg0.with_(router="open_loop")
models = {"base": BASE}

# ---- phase 1: base model on dev, open loop -------------------------------------------------------
proc, gen = serve(BASE)
dev_base = pool("dev_base_openloop", gen, open_loop, dev_items, K_DEV)

# arms 1, 2, 5 and the self-consistency baseline use the base model: run them while it is served
cal_base = calibrator_from(dev_base)
for s in SEEDS:   # base arms have no training seed; the K test runs are repeated per seed slot for pairing
    pool(f"test_base_seed{s}", gen, open_loop, test_items, K_TEST)
    pool(f"test_base_prompt_seed{s}", gen, open_loop.with_(prompt_variant="abstain_strong"), test_items, K_TEST)
    pool(f"test_base_gate_seed{s}", gen, cfg0, test_items, K_TEST, calibrator=cal_base)
pool("test_base_selfconsistency", gen, open_loop, test_items, K_SC)
stop(proc)

# ---- phase 1b: label the dev answers with the cross-family judge, then build the training data ------
# The token scorer penalises correct paraphrases (human audit in gatedrag), which would turn answerable
# questions into "abstain" pairs. Pairs are labelled by the judge (Mistral-7B, a different family).
label = "correct"
if not DRY:
    from gatedrag.evaluation.judge import judge_pool
    judged_path = OUT / "dev_base_openloop_judged.json"
    if not judged_path.exists():
        jproc, jgen = serve(JUDGE, tp=2)
        save_pool(judge_pool(dev_base, dev_items, jgen, concurrency=16), judged_path)
        stop(jproc)
    dev_base = load_pool(judged_path)
    label = "correct_judge"
built = build(dev_base, dev_raw, prompt_builder(str(chunks)), test_ids=test_ids, label=label, max_pairs_per_item=3)
write_jsonl(built.sft, OUT / "sft.jsonl")
write_jsonl(built.pairs, OUT / "pairs.jsonl")
(OUT / "data_counts.json").write_text(json.dumps({**built.counts, "label": label}, indent=1))
log(f"training data ({label}): {built.counts}")

# ---- phases 2-4, one seed at a time ---------------------------------------------------------------
# Merged models (about 6 GB each in fp16) live in MODELS, outside the size-limited run output, and are
# deleted once that seed is evaluated: a first attempt that kept every seed's models in
# /kaggle/working died with "No space left on device". Adapters and run_info (small) are kept in OUT.
MODELS = Path(os.environ.get("P2_MODELS", str(OUT.parent / "models" if DRY else "/tmp/p2_models")))
tcfg = TrainConfig(base_model=TINY if DRY else BASE)
if DRY:
    tcfg.max_steps, tcfg.batch_size, tcfg.grad_accum = 2, 1, 1
    tcfg.target_modules = ["q_proj", "v_proj"]


def lm_eval_side_effects(name: str, path) -> None:
    target = OUT / f"lm_eval_{name}"
    if DRY or target.exists():
        return
    subprocess.run([LMEVAL_PY, "-m", "lm_eval", "--model", "hf", "--model_args", f"pretrained={path},dtype=float16",
                    "--tasks", "arc_easy,hellaswag", "--limit", "500", "--batch_size", "16",
                    "--output_path", str(target)], check=False)


lm_eval_side_effects("base", BASE)
for s in SEEDS:
    if all((OUT / f"{n}_seed{s}.json").exists() for n in ("test_sft", "test_sft_dpo", "test_sft_dpo_gate")):
        continue
    run_dir = MODELS / f"seed{s}"
    cfg_path = OUT / "train_config.json"
    cfg_path.write_text(json.dumps(asdict(tcfg)))
    # its own process: training memory must be fully released before vLLM serves the merged model
    subprocess.run([sys.executable, "-m", "abstaingate.models.train", str(OUT / "sft.jsonl"), str(OUT / "pairs.jsonl"),
                    str(run_dir), str(s), str(cfg_path)], check=True,
                   env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ag_root / "src"), str(gd_root / "src")])})
    info = json.loads((run_dir / "run_info.json").read_text())
    log(f"seed {s} trained: {info['gpu_hours']:.2f} GPU-hours")
    keep = OUT / f"seed{s}"
    keep.mkdir(exist_ok=True)
    shutil.copy(run_dir / "run_info.json", keep / "run_info.json")
    for adapter in ("sft_adapter", "dpo_adapter"):
        shutil.copytree(run_dir / adapter, keep / adapter, dirs_exist_ok=True)
    for arm, sub in (("sft", "sft_merged"), ("sft_dpo", "dpo_merged")):
        proc, gen = serve(str(run_dir / sub))
        pool(f"test_{arm}_seed{s}", gen, open_loop, test_items, K_TEST)
        if arm == "sft_dpo":
            dev_dpo = pool(f"dev_sft_dpo_openloop_seed{s}", gen, open_loop, dev_items, 2)
            pool(f"test_sft_dpo_gate_seed{s}", gen, cfg0, test_items, K_TEST, calibrator=calibrator_from(dev_dpo))
        stop(proc)
    if s == SEEDS[0]:
        lm_eval_side_effects("sft", run_dir / "sft_merged")
        lm_eval_side_effects("sft_dpo", run_dir / "dpo_merged")
    shutil.rmtree(run_dir, ignore_errors=True)
    log(f"seed {s} evaluated; merged models deleted")

# ---- phase 5: contamination + analysis -----------------------------------------------------------
dev_q = {it["id"]: it["question"] for it in dev_raw}
cont = contamination.report([dev_q[r["item"]] for r in built.sft], [g.question for g in test_items])
(OUT / "contamination.json").write_text(json.dumps(cont, indent=1))
res = analyse(OUT, n_boot=200 if DRY else 2000)
(OUT / "report.json").write_text(json.dumps(res, indent=1))
banner = ("DRY RUN: NOT A RESULT\n\n" if DRY else
          "PILOT: golden items are unreviewed LLM drafts (manifest pilot_only); NOT a result.\n\n"
          if manifest.get("pilot_only") else "")
(OUT / "REPORT.md").write_text(banner + render(res)
                               + f"\nContamination: {json.dumps(cont)}\n")
log(f"done; verdict {res['verdict']}")
