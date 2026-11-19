"""QLoRA SFT, then DPO, for one seed.

  SFT  LoRA on the base model, on correct answers and justified abstentions (data/pairs.py).
       The adapter is merged into an fp16 copy of the base (a 4-bit model cannot be merged).
  DPO  a fresh LoRA on the merged SFT model, with type A + type B pairs; the reference model is
       the SFT model itself (the adapter disabled), as in standard SFT -> DPO.

On a CUDA machine the trainable model is loaded in 4-bit NF4 (QLoRA). Without CUDA (dry runs and
CI) it is loaded in fp32 with no quantisation, so the code path is the same except for the loader.
GPU-hours are recorded in run_info.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    sft_lr: float = 2e-4
    sft_epochs: float = 2.0
    dpo_lr: float = 5e-6
    dpo_epochs: float = 1.0
    dpo_beta: float = 0.1
    batch_size: int = 1
    grad_accum: int = 16
    max_length: int = 4096                    # prompts carry 5 passages (~1.5-2.5k tokens); a shorter limit
                                              # truncates the completion and TRL drops the example
    max_steps: int = -1                       # > 0 only for dry runs
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                                               "gate_proj", "up_proj", "down_proj"])


def _load(model_id: str, quantize: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if quantize:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
        # dtype=float16: Qwen's config says bfloat16, which T4s cannot compute in, and bf16 LoRA weights
        # crash fp16 gradient scaling ("_amp_foreach_non_finite_check_and_unscale_cuda" for BFloat16).
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=q, device_map="auto",
                                                     dtype=torch.float16)
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    return model, tok


def _with_lora(model, lora):
    """Attach LoRA ourselves and keep every trainable weight in float32, so the optimiser never sees
    half-precision gradients whatever dtype the base model was loaded in."""
    from peft import get_peft_model

    model = get_peft_model(model, lora)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model


def _merge(base_id: str, adapter_dir: Path, out_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16 if cuda else torch.float32,
                                                device_map="auto" if cuda else None)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(out_dir))
    AutoTokenizer.from_pretrained(base_id).save_pretrained(str(out_dir))
    keep_source_config(base_id, out_dir)


SOURCE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                "merges.txt", "special_tokens_map.json", "added_tokens.json")


def keep_source_config(source_id: str, out_dir: Path) -> None:
    """Copy the SOURCE model's config and tokenizer files over the merged ones. A LoRA merge changes
    neither, but transformers 5 re-serialises both in its own schema, and the older transformers inside
    the vLLM environment misreads them:
      - config: `rope_parameters` instead of `rope_theta` is silently ignored (wrong RoPE base);
      - tokenizer: `extra_special_tokens` saved as a list crashes 4.47 ("'list' object has no attribute
        'keys'"), seen on Kaggle when vLLM tried to serve the merged SFT model."""
    out = Path(out_dir)
    src_dir = Path(source_id)
    if not src_dir.exists():
        from huggingface_hub import snapshot_download
        src_dir = Path(snapshot_download(source_id, allow_patterns=list(SOURCE_FILES)))
    for name in SOURCE_FILES:
        f = src_dir / name
        if f.exists():
            (out / name).write_bytes(f.read_bytes())
    if (src_dir / "tokenizer_config.json").exists() and (out / "chat_template.jinja").exists() \
            and "chat_template" in (src_dir / "tokenizer_config.json").read_text():
        (out / "chat_template.jinja").unlink()     # the original keeps its template inside tokenizer_config


def _conv(rows: list[dict], kind: str):
    from datasets import Dataset

    if kind == "sft":
        return Dataset.from_list([{"prompt": r["prompt"], "completion": [{"role": "assistant", "content": r["completion"]}]}
                                  for r in rows])
    return Dataset.from_list([{"prompt": r["prompt"], "chosen": [{"role": "assistant", "content": r["chosen"]}],
                               "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows])


def train_seed(sft_rows: list[dict], pair_rows: list[dict], out: str | Path, seed: int,
               cfg: TrainConfig | None = None) -> dict:
    import torch
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    cfg = cfg or TrainConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=cfg.target_modules, task_type="CAUSAL_LM")
    common = dict(per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum,
                  seed=seed, data_seed=seed, logging_steps=10, save_strategy="no", report_to=[],
                  max_steps=cfg.max_steps, fp16=False, bf16=False, use_cpu=not cuda, gradient_checkpointing=cuda)
    # no automatic mixed precision: the 4-bit layers already compute in fp16, the LoRA weights are fp32
    info: dict = {"seed": seed, "config": asdict(cfg), "cuda": cuda,
                  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}

    t0 = time.time()
    model, tok = _load(cfg.base_model, quantize=cuda)
    sft = SFTTrainer(model=_with_lora(model, lora), processing_class=tok, train_dataset=_conv(sft_rows, "sft"),
                     args=SFTConfig(output_dir=str(out / "sft_ckpt"), learning_rate=cfg.sft_lr,
                                    num_train_epochs=cfg.sft_epochs, max_length=cfg.max_length,
                                    completion_only_loss=True, **common))
    info["sft_examples"] = len(sft.train_dataset)
    if info["sft_examples"] == 0:
        raise ValueError("every SFT example was dropped (prompt longer than max_length?)")
    info["sft_train"] = sft.train().metrics
    sft.model.save_pretrained(str(out / "sft_adapter"))
    del sft, model
    _merge(cfg.base_model, out / "sft_adapter", out / "sft_merged")
    info["sft_hours"] = (time.time() - t0) / 3600

    t1 = time.time()
    model, tok = _load(str(out / "sft_merged"), quantize=cuda)
    dpo = DPOTrainer(model=_with_lora(model, lora), ref_model=None, processing_class=tok,
                     train_dataset=_conv(pair_rows, "dpo"),
                     args=DPOConfig(output_dir=str(out / "dpo_ckpt"), learning_rate=cfg.dpo_lr, beta=cfg.dpo_beta,
                                    num_train_epochs=cfg.dpo_epochs, max_length=cfg.max_length, **common))
    info["dpo_train"] = dpo.train().metrics
    dpo.model.save_pretrained(str(out / "dpo_adapter"))
    del dpo, model
    _merge(str(out / "sft_merged"), out / "dpo_adapter", out / "dpo_merged")
    info["dpo_hours"] = (time.time() - t1) / 3600
    n_gpu = max(1, len(info["gpus"])) if cuda else 0
    info["gpu_hours"] = (info["sft_hours"] + info["dpo_hours"]) * n_gpu
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    return info


def main(argv: list[str] | None = None) -> None:
    """`python -m abstaingate.models.train SFT.jsonl PAIRS.jsonl OUT SEED [CONFIG.json]`: one seed in its
    own process, so every byte of GPU memory it used is released when it exits. In-process training
    left about 7 GB allocated, and the vLLM server started next ran out of memory (seen on Kaggle)."""
    import sys
    a = argv if argv is not None else sys.argv[1:]
    rows = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]  # noqa: E731
    cfg = TrainConfig(**json.loads(Path(a[4]).read_text())) if len(a) > 4 else TrainConfig()
    train_seed(rows(a[0]), rows(a[1]), a[2], int(a[3]), cfg)


if __name__ == "__main__":
    main()
