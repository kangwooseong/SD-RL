"""Train the depth / size policies with PPO. ViSpec and the target VLM stay frozen.

Steps
  1. calibration: measure plain ViSpec (reference throughput) and fit the cost model
  2. PPO: collect cycles, update the policies, refresh the reference throughput
  3. validation every --val-every updates on held-out prompts (measured speed,
     greedy actions); the best checkpoint is saved as policy_best.pt

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
from .data import NAMES, load_prompts
from .ppo import PPOTrainer


def generate(model, inputs, args, max_new_tokens=None):
    model.specgenerate(**inputs, temperature=args.temperature,
                       max_new_tokens=max_new_tokens or args.max_new_token, log=True)


def calibrate(model, ctrl, pool, args):
    """Reference throughput of plain ViSpec + cost model. Always uses measured time."""
    ctrl.measure, ctrl.log_cycles, ctrl.time_source = True, True, "measured"
    # spread the calibration prompts over all training sets
    n = min(args.calib_prompts, len(pool))
    picks = random.Random(args.seed).sample(range(len(pool)), n)

    def run():
        ctrl.cycle_log = []
        for j in picks:
            generate(model, pool.inputs(j), args)
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


def validate(model, ctrl, pool, args, use_policy):
    """Measured throughput on the validation prompts (first cycle of each prompt excluded).

    use_policy=False: plain ViSpec (--depth / --total-token).
    use_policy=True : greedy actions of the policies trained so far; a policy that has
                      not been trained yet falls back to the fixed ViSpec value.
    Training state (modes, buffers, reward settings) is restored afterwards.
    """
    saved = (ctrl.depth_mode, ctrl.size_mode, ctrl.train_depth, ctrl.train_size,
             ctrl.measure, ctrl.time_source)
    depth_mode = "policy" if use_policy and ctrl.trained["depth"] else "fixed"
    size_mode = "policy" if use_policy and ctrl.trained["size"] else "fixed"
    ctrl.set_modes(depth_mode, size_mode, False, False)
    ctrl.measure, ctrl.time_source = True, "measured"
    ctrl.reset_stats()
    for j in range(len(pool)):
        generate(model, pool.inputs(j), args)
    summ = ctrl.summary()
    ctrl.reset_stats()
    ctrl.depth_mode, ctrl.size_mode, ctrl.train_depth, ctrl.train_size = saved[:4]
    ctrl.measure, ctrl.time_source = saved[4:]
    return summ


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
    # data (see vispec/rl/data.py for the splits)
    p.add_argument("--dataset", type=str, default="coco_caption",
                   help=f"comma separated training sets from {NAMES}")
    p.add_argument("--train-num", type=int, default=2000, help="training prompts per set")
    p.add_argument("--val-dataset", type=str, default=None,
                   help="comma separated validation sets; default: same as --dataset")
    p.add_argument("--val-num", type=int, default=30, help="validation prompts per set (0 = off)")
    p.add_argument("--val-every", type=int, default=10, help="in PPO updates")
    p.add_argument("--llava-pretrain-dir", type=str, default=None,
                   help="folder with blip_laion_cc_sbu_558k.json and its images")
    p.add_argument("--llava-pretrain-skip", type=int, default=68000,
                   help="skip the prompts ViSpec's draft model was trained on")
    p.add_argument("--textvqa-dir", type=str, default=None,
                   help="folder with TextVQA_0.5.1_train.json and train_images/")
    p.add_argument("--scienceqa-dir", type=str, default="vispec/data/scienceqa")
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
    pool = load_prompts(args.dataset, "train", args.train_num, args)
    print(f"{len(pool)} training prompts: {pool.counts()}")
    val_pool = None
    if args.val_num > 0:
        val_pool = load_prompts(args.val_dataset or args.dataset, "val", args.val_num, args)
        print(f"{len(val_pool)} validation prompts: {val_pool.counts()}")

    log_f = open(os.path.join(args.save_dir, "train_log.jsonl"), "a")
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # warm-up (CUDA kernels, caches)
    for _ in range(3):
        generate(model, pool.inputs(0), args, max_new_tokens=32)

    # ---- 1) calibration -------------------------------------------------
    resumed_rate = ctrl.rate_ref
    if args.skip_calib:
        if args.time_source == "model" and not ctrl.cost.fitted:
            raise ValueError("--skip-calib with --time-source model needs a cost model in --resume")
    else:
        calib = calibrate(model, ctrl, pool, args)
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

    # plain ViSpec on the validation prompts: the number every validation is compared to
    val_base, best_val = None, 0.0
    if val_pool is not None:
        val_base = validate(model, ctrl, val_pool, args, use_policy=False)["tokens_per_sec"]
        print(f"validation, plain ViSpec: {val_base:.2f} tokens/sec")
        log_f.write(json.dumps({"val_vispec_tokens_per_sec": val_base}) + "\n")

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

    order = list(range(len(pool)))
    random.shuffle(order)
    i, n_update, cycles_done, t_start = 0, 0, 0, time.time()
    while cycles_done < args.total_cycles:
        if i == len(order):
            random.shuffle(order)
            i = 0
        generate(model, pool.inputs(order[i]), args)
        i += 1
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
            if val_pool is not None and n_update % args.val_every == 0:
                v = validate(model, ctrl, val_pool, args, use_policy=True)
                rec.update({
                    "val_tokens_per_sec": v["tokens_per_sec"],
                    "val_speedup_vs_vispec": v["tokens_per_sec"] / val_base,
                    "val_mean_depth": v["mean_depth"],
                    "val_mean_size": v["mean_size"],
                })
                if v["tokens_per_sec"] > best_val:
                    best_val = v["tokens_per_sec"]
                    ctrl.save(os.path.join(args.save_dir, "policy_best.pt"))
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
