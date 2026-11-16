"""Prompt sets for training / validating the RL policies.

Only prompts (image + question) are needed: the reward comes from speed, not labels.
Every set is split so that it never overlaps with the prompts used by the benchmark
scripts in vispec/evaluation:

  name             train split                           validation split        benchmark uses
  llava_pretrain   seed-42 shuffle, index >= 68000       last 500 of the shuffle  (not a benchmark; ViSpec trained on 0..67999)
  scienceqa        "train" problems with an image        "val" problems           "test"
  textvqa          TextVQA_0.5.1_train.json              last 500 of the list     TextVQA_0.5.1_val.json
  coco_caption     seed-42 shuffle, index >= 100         last 500 of the shuffle  index 0..99

Local data
  llava_pretrain: --llava-pretrain-dir with blip_laion_cc_sbu_558k.json and the images
                  (liuhaotian/LLaVA-Pretrain on Hugging Face)
  textvqa:        --textvqa-dir with TextVQA_0.5.1_train.json and train_images/
  scienceqa / coco_caption are downloaded through `datasets`.
"""

import json
import os
import random
from types import SimpleNamespace

VAL_RESERVE = 500  # prompts at the end of each list that are never used for training
NAMES = ("llava_pretrain", "scienceqa", "textvqa", "coco_caption")


class PromptPool:
    """Flat list of (dataset name, item, builder). inputs(i) builds the model inputs."""

    def __init__(self):
        self.items = []

    def add(self, name, items, builder):
        self.items += [(name, it, builder) for it in items]

    def __len__(self):
        return len(self.items)

    def inputs(self, i):
        _, item, builder = self.items[i]
        return builder(item)

    def counts(self):
        out = {}
        for name, _, _ in self.items:
            out[name] = out.get(name, 0) + 1
        return out


def _take(n_total, split, num, train_start=0):
    """Index range for a split: train from the front, validation from the reserved tail."""
    if split == "val":
        return range(max(n_total - VAL_RESERVE, 0), n_total)[:num]
    return range(train_start, max(n_total - VAL_RESERVE, train_start))[:num]


def _llava_pretrain(split, num, args):
    from datasets import Dataset
    from PIL import Image

    from ..evaluation.coco_caption_prompt import build_prompt

    root = args.llava_pretrain_dir
    if not root:
        raise ValueError("llava_pretrain needs --llava-pretrain-dir")
    with open(os.path.join(root, "blip_laion_cc_sbu_558k.json")) as f:
        ds = Dataset.from_list(json.load(f)).shuffle(seed=42)  # same order as ViSpec's ge_data
    idx = _take(len(ds), split, num, train_start=args.llava_pretrain_skip)
    items = [os.path.join(root, ds[i]["image"]) for i in idx]

    def builder(path):
        # detailed-description prompt -> long answers, i.e. many cycles per prompt
        return build_prompt({"image": Image.open(path).convert("RGB")}, args)

    return items, builder


def _scienceqa(split, num, args):
    from datasets import load_dataset

    from ..evaluation.scienceqa_prompt import build_prompt

    root = args.scienceqa_dir
    problems = json.load(open(os.path.join(root, "problems.json")))
    pid_splits = json.load(open(os.path.join(root, "pid_splits.json")))
    hf_split = "train" if split == "train" else "validation"
    hf = load_dataset("derek-thomas/ScienceQA")[hf_split]
    # problems.json and the HF split are in the same order (as in gen_spec_answer_sqa.py)
    pairs = [
        (qid, i)
        for i, qid in enumerate(pid_splits["train" if split == "train" else "val"])
        if problems[qid]["image"] is not None
    ]
    random.Random(42).shuffle(pairs)
    pairs = pairs[:num]
    for qid, _ in pairs:
        problems[qid]["caption"] = ""
    # same prompt settings as exp.sh (zero-shot, QCM-ALE)
    sqa_args = SimpleNamespace(
        model=args.model, use_caption=False, options=["A", "B", "C", "D", "E"], prompt_format="QCM-ALE"
    )

    def builder(pair):
        qid, i = pair
        problems[qid]["image"] = hf[i]["image"]
        inputs = build_prompt(problems, [], qid, sqa_args)
        problems[qid]["image"] = True  # drop the decoded image again
        return inputs

    return pairs, builder


def _textvqa(split, num, args):
    from PIL import Image

    from ..evaluation.textvqa_prompt import build_prompt

    root = args.textvqa_dir
    if not root:
        raise ValueError("textvqa needs --textvqa-dir")
    data = json.load(open(os.path.join(root, "TextVQA_0.5.1_train.json")))["data"]
    random.Random(42).shuffle(data)
    items = [
        (os.path.join(root, "train_images", data[i]["image_id"] + ".jpg"), data[i]["question"])
        for i in _take(len(data), split, num)
    ]

    def builder(item):
        path, question = item
        return build_prompt({"image": Image.open(path).convert("RGB"), "question": question}, args)

    return items, builder


def _coco_caption(split, num, args):
    from datasets import load_dataset

    from ..evaluation.coco_caption_prompt import build_prompt

    dataset = load_dataset("HuggingFaceM4/COCO", split="test")
    imgid_indices = {d["imgid"]: idx for idx, d in enumerate(dataset)}
    dataset = dataset.select(imgid_indices.values()).shuffle(seed=42)  # as in the benchmark script
    items = list(_take(len(dataset), split, num, train_start=100))

    def builder(i):
        return build_prompt(dataset[i], args)

    return items, builder


_LOADERS = {
    "llava_pretrain": _llava_pretrain,
    "scienceqa": _scienceqa,
    "textvqa": _textvqa,
    "coco_caption": _coco_caption,
}


def load_prompts(names, split, num_per_set, args):
    """names: comma separated subset of NAMES. Returns a PromptPool."""
    pool = PromptPool()
    for name in [n.strip() for n in names.split(",") if n.strip()]:
        if name not in _LOADERS:
            raise ValueError(f"unknown dataset '{name}', choose from {NAMES}")
        items, builder = _LOADERS[name](split, num_per_set, args)
        if not items:
            raise ValueError(f"dataset '{name}' ({split}) is empty")
        pool.add(name, items, builder)
    return pool
