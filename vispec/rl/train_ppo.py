"""Train the depth / size policies with PPO. ViSpec and the target VLM stay frozen.

Steps
  1. calibration: measure plain ViSpec (reference throughput) and fit the cost model
  2. PPO: collect cycles, update the policies, refresh the reference throughput

Example (full recipe in README.md):
python -m vispec.rl.train_ppo \
  --base-model-path llava-hf/llava-v1.6-vicuna-7b-hf \
  --spec-model-path JLKang/ViSpec-llava-v1.6-vicuna-7b-hf \
  --train size --depth-mode random --save-dir rl_ckpt/s1
"""

import argparse
import json
import os
import random
import time

import torch

from .controller import DraftController, RLConfig
from .ppo import PPOTrainer


def load_train_data(args):
    """Prompts used to train the policies. Must not overlap with the evaluation prompts.

    Only prompts (image + question) are needed: the reward comes from speed, not labels.
    The ViSpec COCO-caption benchmark uses the first 100 images of the seed-42 shuffle,
    so training starts at index --train-start (default 100) of the same shuffle.
    Add other training sets here (return an indexable dataset + its build_prompt).
    """
    if args.dataset == "coco_caption":
        from datasets import load_dataset

        from ..evaluation.coco_caption_prompt import build_prompt

        dataset = load_dataset("HuggingFaceM4/COCO", split="test")
        imgid_indices = {d["imgid"]: idx for idx, d in enumerate(dataset)}
        dataset = dataset.select(imgid_indices.values()).shuffle(seed=42)
        end = min(args.train_start + args.train_num, len(dataset))
        if args.train_start < 100:
            print("WARNING: --train-start < 100 overlaps with the COCO-caption benchmark prompts")
        return dataset.select(range(args.train_start, end)), build_prompt
    raise NotImplementedError(f"unknown --dataset {args.dataset}")


def calibrate(model, ctrl, data, build_prompt, args):
    """Reference throughput of plain ViSpec + cost model. Always uses measured time."""
    ctrl.measure, ctrl.log_cycles, ctrl.time_source = True, True, "measured"
    n = min(args.calib_prompts, len(data))

    def run():
        ctrl.cycle_log = []
        for j in range(n):
            model.specgenerate(**build_prompt(data[j], args), temperature=args.temperature,
                               max_new_tokens=args.max_new_token, log=True)
        return ctrl.cycle_log

    # A) ViSpec's fixed setting (--depth / --total-token)
    ctrl.set_modes("fixed", "fixed", False, False)
    rec_fixed = run()
    # B) random depth / size for coverage; the policy nets are evaluated so that
    #    their inference cost is part of the fitted draft time
    ctrl.set_modes("random", "random", False, False)
    ctrl.calib_policy_cost = True
    rec_rand = run()
    ctrl.calib_policy_cost, ctrl.log_cycles, ctrl.cycle_log = False, False, []

    fit = ctrl.cost.fit(rec_fixed + rec_rand)
    tok = sum(r[4] for r in rec_fixed)
    rate_measured = tok / sum(r[5] + r[6] for r in rec_fixed)
    rate_model = tok / sum(ctrl.cost.cycle(r[0], r[1], r[2], r[3]) for r in rec_fixed)
    ctrl.reset_stats()
    return {**fit, "vispec_rate_measured": rate_measured, "vispec_rate_model": rate_model}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-path", type=str, required=True)
    p.add_argument("--spec-model-path", type=str, required=True)
    p.add_argument("--num-q", type=int, default=2)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--depth", type=int, default=3, help="ViSpec depth (fixed mode / calibration)")
    p.add_argument("--total-token", type=int, default=30, help="ViSpec size (fixed mode / calibration)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-token", type=int, default=256)
    p.add_argument("--dataset", type=str, default="coco_caption")
    p.add_argument("--train-start", type=int, default=100)
    p.add_argument("--train-num", type=int, default=2000)
    # what to train / how the other decision is taken
    p.add_argument("--train", choices=["depth", "size", "both"], default="both")
    p.add_argument("--depth-mode", choices=["fixed", "random", "policy"], default=None)
    p.add_argument("--size-mode", choices=["fixed", "random", "policy"], default=None)
    p.add_argument("--resume", type=str, default=None, help="checkpoint to start from")
    # policy / environment
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-depth", type=int, default=1)
    p.add_argument("--size-choices", type=str, default="10,20,30,40,50,60")
    p.add_argument("--no-visual", action="store_true", help="ablation: drop g_v and image ratio")
    # reward
    p.add_argument("--reward-mode", choices=["gain", "throughput"], default="gain",
                   help="gain: tokens - rate_ref*time (recommended); throughput: tokens/time per cycle")
    p.add_argument("--time-source", choices=["measured", "model"], default="measured",
                   help="model: use the fitted cost model instead of noisy wall-clock time")
    p.add_argument("--gamma", type=float, default=0.97, help="discount between cycles (gain mode)")
    p.add_argument("--gae-lambda", type=float, default=0.9)
    p.add_argument("--rate-init", type=float, default=None,
                   help="reference throughput in tokens/sec; default: plain ViSpec from calibration")
    p.add_argument("--rate-ema", type=float, default=0.5, help="refresh speed of the reference throughput")
    p.add_argument("--calib-prompts", type=int, default=30)
    p.add_argument("--skip-calib", action="store_true",
                   help="reuse cost model / rate from --resume (same GPU and model only)")
    # PPO
    p.add_argument("--total-cycles", type=int, default=100000)
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--save-every", type=int, default=10, help="in PPO updates")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.model = args.base_model_path  # build_prompt reads args.model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    from ..model.spec_model_ours import SpecModel

    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        num_q=args.num_q,
    )
    model.eval()
    device = model.spec_layer.embed_tokens.weight.device

    if args.resume:
        ctrl = DraftController.load(args.resume, device)
        ctrl.cfg.reward_mode = args.reward_mode
        ctrl.cfg.gamma, ctrl.cfg.gae_lambda = args.gamma, args.gae_lambda
    else:
        ctrl = DraftController(
            RLConfig(
                top_k=args.top_k,
                hidden_size=model.hidden_size,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                size_choices=[int(x) for x in args.size_choices.split(",")],
                use_visual=not args.no_visual,
                reward_mode=args.reward_mode,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            ),
            device,
        )
    ctrl.attach(model.spec_layer)
    data, build_prompt = load_train_data(args)
    print(f"{len(data)} training prompts")

    log_f = open(os.path.join(args.save_dir, "train_log.jsonl"), "a")
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # warm-up (CUDA kernels, caches)
    for _ in range(3):
        model.specgenerate(**build_prompt(data[0], args), temperature=args.temperature,
                           max_new_tokens=32, log=True)

    # ---- 1) calibration -------------------------------------------------
    resumed_rate = ctrl.rate_ref
    if args.skip_calib:
        if args.time_source == "model" and not ctrl.cost.fitted:
            raise ValueError("--skip-calib with --time-source model needs a cost model in --resume")
    else:
        calib = calibrate(model, ctrl, data, build_prompt, args)
        print("calibration:", json.dumps(calib))
        log_f.write(json.dumps({"calibration": calib, "cost": ctrl.cost.state()}) + "\n")
        if resumed_rate is None:
            ctrl.rate_ref = calib["vispec_rate_model" if args.time_source == "model"
                                  else "vispec_rate_measured"]
    if args.rate_init is not None:
        ctrl.rate_ref = args.rate_init
    if args.reward_mode == "gain" and ctrl.rate_ref is None:
        raise ValueError("gain reward needs a reference throughput: run calibration or pass --rate-init")
    print(f"reference throughput: {ctrl.rate_ref} tokens/sec")

    # ---- 2) PPO -----------------------------------------------------------
    train_depth = args.train in ("depth", "both")
    train_size = args.train in ("size", "both")
    depth_mode = args.depth_mode or ("policy" if train_depth or ctrl.trained["depth"] else "fixed")
    size_mode = args.size_mode or ("policy" if train_size or ctrl.trained["size"] else "fixed")
    ctrl.set_modes(depth_mode, size_mode, train_depth, train_size)
    ctrl.time_source = args.time_source
    ctrl.measure = args.time_source == "measured"
    ctrl.buf = {"depth": [], "size": []}
    ctrl.reset_stats()
    ctrl.update_rate(0.0)  # clear rollout counters, keep rate_ref
    trainer = PPOTrainer(
        ctrl, lr=args.lr, epochs=args.epochs, minibatch=args.minibatch, ent_coef=args.ent_coef
    )
    print(f"depth: mode={ctrl.depth_mode} train={ctrl.train_depth} | "
          f"size: mode={ctrl.size_mode} train={ctrl.train_size} | "
          f"reward={args.reward_mode} time={args.time_source}")

    order = list(range(len(data)))
    random.shuffle(order)
    i, n_update, cycles_done, t_start = 0, 0, 0, time.time()
    while cycles_done < args.total_cycles:
        if i == len(order):
            random.shuffle(order)
            i = 0
        inputs = build_prompt(data[order[i]], args)
        i += 1
        model.specgenerate(**inputs, temperature=args.temperature,
                           max_new_tokens=args.max_new_token, log=True)
        ctrl.finish_episode()

        if trainer.num_ready() >= args.rollout:
            summ = ctrl.summary()
            cycles_done += summ["cycles"]
            ctrl.reset_stats()
            stats = trainer.update()
            rate_used = ctrl.rate_ref
            if args.reward_mode == "gain":
                ctrl.update_rate(args.rate_ema)  # once per update: fixed inside a rollout
            else:
                ctrl.update_rate(0.0)
            n_update += 1
            rec = {"update": n_update, "cycles": cycles_done,
                   "elapsed_min": round((time.time() - t_start) / 60, 1),
                   "rate_ref": rate_used, **summ, **stats}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in rec.items()}))
            if n_update % args.save_every == 0:
                ctrl.save(os.path.join(args.save_dir, f"policy_{cycles_done}.pt"))
                ctrl.save(os.path.join(args.save_dir, "policy_last.pt"))
    ctrl.save(os.path.join(args.save_dir, "policy_last.pt"))
    print("saved", os.path.join(args.save_dir, "policy_last.pt"))


if __name__ == "__main__":
    main()
