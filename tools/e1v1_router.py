#!/usr/bin/env python
"""E1 v1 — trained router with frozen embeddings + 5-fold CV (ADAPTIVE_SELECTION.md).

Formulation: per-expert correctness predictors p_e(question, input images); route =
argmax_e p_e. Labels come from the E0 per-question correctness in
router_task_<bench>.json. 5-fold CV within each benchmark (train on 4 folds, route the
held-out fold), so no question routes itself.

Subcommands:
  embed  — GPU: SigLIP-large image embeddings (mean over input views) + MiniLM text
           embeddings -> router_feats_<bench>.npz
  cv     — CPU: logistic heads (torch), 5-fold CV, routed accuracy vs best-single/oracle.
           Variants: text-only / image-only / text+image features; per-benchmark and
           pooled (one router trained on all benchmarks' train folds).

Image sources per benchmark follow the eval scripts (see repo inference scripts):
COSMIC user_1/user_2 paths; MMSI/STARE/BLINK parquet-embedded bytes; MindCube relative
paths; OmniSpatial <root>/<task_type>/<id-prefix>.png.
"""

import argparse
import json
import os
import sys
from io import BytesIO

import numpy as np

E0DIR = "/path/to/scratch/VisualCoT/eval_results/adaptive_selection_e0"
COSMIC_DIR = "/path/to/scratch/VisualCoT/spatial_collab_dataset"
MMSI_PARQUET = "/path/to/scratch/datasets/MMSI-Bench/MMSI_Bench.parquet"
MINDCUBE_DIR = "/path/to/scratch/datasets/MindCube/data"
OMNI_DIR = "/path/to/scratch/datasets/OmniSpatial/OmniSpatial-test"
STARE_PARQUET = "/path/to/scratch/datasets/STARE/perspective/test-00000-of-00001.parquet"
BLINK_PARQUET = "/path/to/scratch/datasets/BLINK/Multi-view_Reasoning/val-00000-of-00001.parquet"

COSMIC = ["anchor", "counting", "relative_distance", "relative_direction"]
ALL_BENCHES = COSMIC + ["mmsi", "mindcube", "omnispatial", "stare", "blink"]
MAX_IMAGES = 6  # MMSI can have up to 10 input images; cap for embedding cost


# ---------------------------------------------------------------- image loading

def _cosmic_loader(bench):
    with open(f"{COSMIC_DIR}/approved_mcqs_{bench}_normalized.json") as f:
        recs = {r["sample_id"]: r for r in json.load(f)}

    def load(sid):
        r = recs[sid]
        paths = [r["user_1_image_local_path"], r["user_2_image_local_path"]]
        return [p.replace("/path/to/scratch",
                          "/path/to/scratch") for p in paths]
    return load


def _mmsi_loader():
    import pandas as pd
    df = pd.read_parquet(MMSI_PARQUET)
    rows = {str(r["id"]): r["images"] for _, r in df.iterrows()}

    def load(sid):
        return [BytesIO(b) for b in rows[sid][:MAX_IMAGES]]
    return load


def _mindcube_loader():
    recs = {}
    with open(f"{MINDCUBE_DIR}/raw/MindCube_tinybench.jsonl") as f:
        for line in f:
            r = json.loads(line)
            recs[r["id"]] = r["images"]

    def load(sid):
        return [os.path.join(MINDCUBE_DIR, rel) for rel in recs[sid][:MAX_IMAGES]]
    return load


def _omni_loader():
    def load(sid):
        task_type, rid = sid.split("|", 1)
        return [os.path.join(OMNI_DIR, task_type, f"{rid.split('_')[0]}.png")]
    return load


def _parquet_bytes_loader(path, key_col, img_cols=None, list_col=None):
    import pandas as pd
    df = pd.read_parquet(path)
    rows = {str(r[key_col]): r for _, r in df.iterrows()}

    def load(sid):
        r = rows[sid]
        if list_col:
            return [BytesIO(x["bytes"]) for x in r[list_col]]
        return [BytesIO(r[c]["bytes"]) for c in img_cols if r[c] is not None]
    return load


def get_loader(bench):
    if bench in COSMIC:
        return _cosmic_loader(bench)
    if bench == "mmsi":
        return _mmsi_loader()
    if bench == "mindcube":
        return _mindcube_loader()
    if bench == "omnispatial":
        return _omni_loader()
    if bench == "stare":
        return _parquet_bytes_loader(STARE_PARQUET, "qid", list_col="images")
    if bench == "blink":
        return _parquet_bytes_loader(BLINK_PARQUET, "idx", img_cols=["image_1", "image_2"])
    raise ValueError(bench)


# ---------------------------------------------------------------- embed

def cmd_embed(args):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, SiglipModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sig = SiglipModel.from_pretrained("google/siglip-large-patch16-384").to(dev).eval()
    sig_proc = AutoImageProcessor.from_pretrained("google/siglip-large-patch16-384")
    txt_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    txt_mod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(dev).eval()

    @torch.no_grad()
    def embed_images(images):
        pil = []
        for im in images:
            pil.append(Image.open(im).convert("RGB") if not isinstance(im, Image.Image) else im)
        feats = []
        for i in range(0, len(pil), 16):
            inp = sig_proc(images=pil[i:i + 16], return_tensors="pt").to(dev)
            feats.append(sig.get_image_features(**inp).float().cpu())
        return torch.cat(feats).numpy()

    @torch.no_grad()
    def embed_texts(texts):
        out = []
        for i in range(0, len(texts), 64):
            enc = txt_tok(texts[i:i + 64], padding=True, truncation=True, max_length=256,
                          return_tensors="pt").to(dev)
            h = txt_mod(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1)).float().cpu())
        return torch.cat(out).numpy()

    for bench in (args.benchmarks or ALL_BENCHES):
        out_fp = f"{E0DIR}/router_feats_{bench}.npz"
        if os.path.exists(out_fp) and not args.force:
            print(f"{bench}: exists, skip")
            continue
        with open(f"{E0DIR}/router_task_{bench}.json") as f:
            rows = json.load(f)
        loader = get_loader(bench)
        sids, img_embs, texts, n_imgs = [], [], [], []
        for k, r in enumerate(rows):
            sid = r["sample_id"]
            try:
                e = embed_images(loader(sid))
            except Exception as ex:
                print(f"  {bench} {sid}: IMAGE FAIL {ex}", file=sys.stderr)
                continue
            sids.append(sid)
            img_embs.append(e.mean(0))
            n_imgs.append(len(e))
            texts.append(str(r["question"])[:2000])
            if (k + 1) % 200 == 0:
                print(f"  {bench}: {k + 1}/{len(rows)}")
        txt_embs = embed_texts(texts)
        np.savez(out_fp, sids=np.array(sids), img=np.stack(img_embs),
                 txt=txt_embs, n_imgs=np.array(n_imgs))
        print(f"{bench}: wrote {len(sids)}/{len(rows)} -> {out_fp}")


# ---------------------------------------------------------------- cv

def _fit_logistic(X, y, wd, iters=300, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(X.shape[1], 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logit = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt) \
            + wd * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy()


def _predict(X, wb):
    w, b = wb
    return 1.0 / (1.0 + np.exp(-(X @ w + b))).ravel()


def _standardize(train, *others):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [(a - mu) / sd for a in (train,) + others]


def cmd_cv(args):
    rng = np.random.RandomState(0)
    WDS = [1e-4, 1e-3, 1e-2]
    variants = {"txt": lambda d: d["txt"], "img": lambda d: d["img"],
                "txt+img": lambda d: np.concatenate([d["txt"], d["img"]], 1)}

    # load everything
    bench_data = {}
    for bench in (args.benchmarks or ALL_BENCHES):
        feats = np.load(f"{E0DIR}/router_feats_{bench}.npz", allow_pickle=True)
        with open(f"{E0DIR}/router_task_{bench}.json") as f:
            task = {r["sample_id"]: r for r in json.load(f)}
        sids = [s for s in feats["sids"] if s in task]
        keep = [i for i, s in enumerate(feats["sids"]) if s in task]
        experts = sorted(task[sids[0]]["correct_by_expert"])
        Y = np.array([[task[s]["correct_by_expert"][e] for e in experts] for s in sids], float)
        bench_data[bench] = {"sids": sids, "experts": experts, "Y": Y,
                             "txt": feats["txt"][keep], "img": feats["img"][keep],
                             "folds": rng.permutation(len(sids)) % args.folds}

    report = {}
    for vname, fx in variants.items():
        print(f"\n==== features: {vname} ====")
        print(f"{'bench':<20}{'n':>6}{'best-single':>12}{'routed':>9}{'pooled':>9}{'oracle':>8}")
        report[vname] = {}
        for bench, d in bench_data.items():
            X, Y, folds = fx(d), d["Y"], d["folds"]
            experts = d["experts"]
            routed = np.zeros(len(X), bool)
            pooled_routed = np.zeros(len(X), bool)
            for f in range(args.folds):
                tr, te = folds != f, folds == f
                Xtr, Xte = _standardize(X[tr], X[te])
                # inner split for weight-decay choice (by routed acc on inner val)
                inner = np.arange(tr.sum()) % 5 != 0
                best_wd, best_score = WDS[0], -1
                for wd in WDS:
                    heads = [_fit_logistic(Xtr[inner], Y[tr][inner, j], wd) for j in range(len(experts))]
                    P = np.stack([_predict(Xtr[~inner], h) for h in heads], 1)
                    score = Y[tr][~inner][np.arange((~inner).sum()), P.argmax(1)].mean()
                    if score > best_score:
                        best_wd, best_score = wd, score
                heads = [_fit_logistic(Xtr, Y[tr][:, j], best_wd) for j in range(len(experts))]
                P = np.stack([_predict(Xte, h) for h in heads], 1)
                routed[te] = Y[te][np.arange(te.sum()), P.argmax(1)] > 0.5

                # pooled: train on all benches' train folds restricted to shared experts
                pool_X, pool_Y = [], []
                for ob, od in bench_data.items():
                    otr = od["folds"] != f
                    cols = [od["experts"].index(e) for e in experts if e in od["experts"]]
                    if len(cols) != len(experts):
                        continue
                    pool_X.append(fx(od)[otr])
                    pool_Y.append(od["Y"][otr][:, cols])
                pX, pY = np.concatenate(pool_X), np.concatenate(pool_Y)
                pXs, Xte_p = _standardize(pX, X[te])
                heads = [_fit_logistic(pXs, pY[:, j], 1e-3) for j in range(len(experts))]
                P = np.stack([_predict(Xte_p, h) for h in heads], 1)
                pooled_routed[te] = Y[te][np.arange(te.sum()), P.argmax(1)] > 0.5

            single = Y.mean(0) * 100
            res = {"n": len(X), "experts": experts,
                   "best_single": float(single.max()),
                   "best_single_expert": experts[int(single.argmax())],
                   "routed": float(routed.mean() * 100),
                   "pooled_routed": float(pooled_routed.mean() * 100),
                   "oracle": float((Y.max(1) > 0.5).mean() * 100)}
            report[vname][bench] = res
            print(f"{bench:<20}{res['n']:>6}{res['best_single']:>12.2f}{res['routed']:>9.2f}"
                  f"{res['pooled_routed']:>9.2f}{res['oracle']:>8.2f}")

    with open(f"{E0DIR}/e1v1_router_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nreport: {E0DIR}/e1v1_router_report.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("embed")
    p1.add_argument("--benchmarks", nargs="*", default=None)
    p1.add_argument("--force", action="store_true")
    p2 = sub.add_parser("cv")
    p2.add_argument("--benchmarks", nargs="*", default=None)
    p2.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    {"embed": cmd_embed, "cv": cmd_cv}[args.cmd](args)
