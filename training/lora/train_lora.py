#!/usr/bin/env python3
"""LoRA fine-tune of MiniCPM5-2B on snap decision prompts.

The supervision is a softmax over the *legal letter tokens only*: each
exported row carries the byte-exact rendered prompt (chat template already
applied by `snap export-prompts`), the ordered letter alphabet for its
question, and a per-letter target distribution (teacher-anchored soft or
smoothed one-hot on the verified label). The model's job is one forward
pass -> read the final-position logits -> softmax over that case's letter
tokens. Nothing else is scored, so no prose generation is ever trained.

Metrics reported per eval pass: argmax accuracy vs `gold_letter`, ECE over
the letter softmax (15 bins), mean CE, and an abstention-inflation probe —
accuracy on abstain-enabled rows that have a committed gold answer, which
collapses when `__abstain__` becomes an attractor.
"""
import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

BASE = "openbmb/MiniCPM5-2B"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def device_auto():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_rows(rows, tok):
    """prompt -> input_ids; letters -> token ids; target -> prob vector.

    Prefer `token_ids` exported by snap (serve-time tokenization: control
    tokens only in the frame, never in the body). Retokenizing the string
    with HF is the fallback — and what the parity check validates."""
    out = []
    for r in rows:
        ids = r.get("token_ids") or tok(r["prompt"], add_special_tokens=False)["input_ids"]
        lidx = [LETTERS.index(l) for l in r["letters"]]
        tgt = [r["target"].get(r["letters"][i], 0.0) for i in range(len(lidx))]
        s = sum(tgt)
        if s <= 0:
            continue
        tgt = [t / s for t in tgt]
        out.append({
            "ids": ids,
            "lidx": lidx,
            "tgt": tgt,
            "gold": LETTERS.index(r["gold_letter"]),
            "abstain": r["abstain"],
            # the abstain slot always sits last when enabled — a committed
            # gold is what the inflation probe measures
            "gold_is_abstain": r["abstain"]
                and r["gold_letter"] == r["letters"][-1],
            "qtype": r["qtype"],
        })
    return out


def batches(rows, bsz, rng):
    """Length-grouped batching: sort, chunk, shuffle chunk order, shuffle
    within chunk — keeps padding waste low without losing randomness."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    chunks = [order[i:i + bsz * 8] for i in range(0, len(order), bsz * 8)]
    for c in chunks:
        rng.shuffle(c)
    rng.shuffle(chunks)
    for c in chunks:
        for i in range(0, len(c), bsz):
            yield [rows[j] for j in c[i:i + bsz]]


def collate(rows, pad_id):
    mx = max(len(r["ids"]) for r in rows)
    ids = torch.full((len(rows), mx), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), mx), dtype=torch.long)
    last = []
    for i, r in enumerate(rows):
        n = len(r["ids"])
        ids[i, :n] = torch.tensor(r["ids"])
        mask[i, :n] = 1
        last.append(n - 1)
    return ids, mask, torch.tensor(last)


def trunk_and_head(model):
    """(decoder trunk, lm_head) under/around a peft wrapper — lets us run
    lm_head only on each row's real last position instead of B x seq x V
    (on MiniCPM's ~150k vocab that projection alone OOMs shared memory)."""
    lm = model.base_model.model if hasattr(model, "base_model") else model
    return lm.model, lm.lm_head


def forward_letters(trunk, lm_head, ids, mask, last, variant_ids, letter_ix):
    """(B, 26) letter logits = per-letter max over single-token variant
    logits — same pooling snap's `read_row` applies at serve time."""
    hidden = trunk(input_ids=ids, attention_mask=mask).last_hidden_state
    bl = lm_head(hidden[torch.arange(ids.size(0)), last])  # (B, V)
    vl = bl[:, variant_ids]                                # (B, nvars)
    g = vl[:, letter_ix.clamp(min=0)]                      # (B, 26, maxv)
    g = g.masked_fill((letter_ix < 0).unsqueeze(0), float("-inf"))
    return g.max(-1).values                                # (B, 26)


def row_loss(bl, rows):
    """Sum of per-row CE over that row's legal letters."""
    losses = []
    for i, r in enumerate(rows):
        l = bl[i, r["lidx"]]                              # (n,)
        tgt = torch.tensor(r["tgt"], dtype=l.dtype, device=l.device)
        losses.append(-(tgt * F.log_softmax(l, dim=-1)).sum())
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, rows, variant_ids, letter_ix, pad_id, dev, bsz, bins=15):
    model.eval()
    trunk, lm_head = trunk_and_head(model)
    n = correct = 0
    ce_sum = 0.0
    ab_n = ab_ok = 0
    acc_b = [0.0] * bins
    conf_b = [0.0] * bins
    cnt_b = [0] * bins
    for i in range(0, len(rows), bsz):
        batch = rows[i:i + bsz]
        ids, mask, last = collate(batch, pad_id)
        ids, mask, last = ids.to(dev), mask.to(dev), last.to(dev)
        bl = forward_letters(trunk, lm_head, ids, mask, last,
                             variant_ids, letter_ix)
        for j, r in enumerate(batch):
            l = bl[j, r["lidx"]]
            p = F.softmax(l, dim=-1)
            tgt = torch.tensor(r["tgt"], dtype=p.dtype, device=p.device)
            ce_sum += float(-(tgt * p.log()).sum())
            am = int(p.argmax())
            hit = am == r["gold"]
            n += 1
            correct += hit
            conf = float(p.max())
            b = min(bins - 1, int(conf * bins))
            cnt_b[b] += 1
            acc_b[b] += float(hit)
            conf_b[b] += conf
            if r["abstain"] and not r["gold_is_abstain"]:
                ab_n += 1
                ab_ok += hit
    ece = sum(abs(acc_b[b] - conf_b[b])
              for b in range(bins) if cnt_b[b]) / max(1, n)
    model.train()
    return {
        "n": n,
        "acc": correct / max(1, n),
        "ece": ece,
        "ce": ce_sum / max(1, n),
        "abstain_committed_acc": (ab_ok / ab_n if ab_n else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True, help="adapter output dir (create-only)")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8, help="grad accumulation -> eff. batch")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default=device_auto())
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--eval-rows", type=int, default=0,
                    help="cap the dev subset used for periodic evals "
                         "(0 = full dev every time)")
    ap.add_argument("--brier", type=float, default=0.0,
                    help="weight of an auxiliary Brier term on the letter "
                         "softmax (0 = pure cross-entropy)")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test row cap")
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists (create-only)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # snap pools each letter over its single-token variants ("A", " A",
    # "\nA") and takes the MAX logit per letter — mirror that exactly, or
    # the trained surface differs from the served one
    letter_variants = []
    variant_ids = []
    for ch in LETTERS:
        ids = []
        for s in (ch, f" {ch}", f"\n{ch}"):
            enc = tok(s, add_special_tokens=False)["input_ids"]
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        if not ids:
            sys.exit(f"letter {ch!r} has no single-token variant in this "
                     "tokenizer — letter alphabet must match llama.cpp")
        for t in ids:
            if t not in variant_ids:
                variant_ids.append(t)
        letter_variants.append([variant_ids.index(t) for t in ids])
    variant_ids = torch.tensor(variant_ids, device=args.device)
    maxv = max(len(v) for v in letter_variants)
    letter_ix = torch.full((len(LETTERS), maxv), -1)
    for L, v in enumerate(letter_variants):
        letter_ix[L, :len(v)] = torch.tensor(v)
    letter_ix = letter_ix.to(args.device)

    train_rows = encode_rows(load_jsonl(args.train), tok)
    dev_rows = encode_rows(load_jsonl(args.dev), tok)
    if args.limit:
        train_rows, dev_rows = train_rows[:args.limit], dev_rows[:args.limit // 5 or 20]
    print(f"train={len(train_rows)} dev={len(dev_rows)} device={args.device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    lconf = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lconf)
    if args.grad_ckpt:
        # PEFT: grads must be required at the model input for ckpt to flow
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    # total_steps counts OPTIMIZER steps; micro counts micro-batches.
    # Conflating them (the old bug) stops the run after 1/accum of an epoch
    # and leaves the cosine schedule stuck near peak LR.
    micro_per_epoch = math.ceil(len(train_rows) / args.bsz)
    opt_per_epoch = math.ceil(micro_per_epoch / args.accum)
    total_steps = max(1, int(opt_per_epoch * args.epochs))
    warm = max(10, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(
        out_dir, f"train-log-{os.path.basename(args.out)}.jsonl"), "x")
    trunk, lm_head = trunk_and_head(model)
    best = -1.0
    micro = opt_step = seen = 0
    done = False
    # periodic evals run on a fixed dev subset when --eval-rows is set —
    # the full dev set would add a large constant to every checkpoint
    eval_pool = dev_rows
    if args.eval_rows and args.eval_rows < len(dev_rows):
        eval_pool = random.Random(0).sample(dev_rows, args.eval_rows)

    def checkpoint():
        nonlocal best
        m = evaluate(model, eval_pool, variant_ids, letter_ix,
                     pad_id, args.device, args.bsz)
        rec = {"opt_step": opt_step, "epoch": epoch, **m}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print("eval", rec, flush=True)
        if m["acc"] > best:
            best = m["acc"]
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for batch in batches(train_rows, args.bsz, rng):
            ids, mask, last = collate(batch, pad_id)
            ids, mask, last = ids.to(args.device), mask.to(args.device), last.to(args.device)
            bl = forward_letters(trunk, lm_head, ids, mask, last,
                                 variant_ids, letter_ix)
            loss = row_loss(bl, batch)
            if args.brier > 0:
                br = []
                for i, r in enumerate(batch):
                    l = bl[i, r["lidx"]]
                    pr = F.softmax(l, dim=-1)
                    tgt = torch.tensor(r["tgt"], dtype=l.dtype, device=l.device)
                    br.append(((pr - tgt) ** 2).sum())
                loss = loss + args.brier * torch.stack(br).mean()
            loss = loss / args.accum
            loss.backward()
            micro += 1
            seen += len(batch)
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 5 == 0:
                    print(f"opt_step {opt_step}/{total_steps} seen={seen} "
                          f"loss={loss.item() * args.accum:.4f}", flush=True)
                if opt_step % 25 == 0 or opt_step >= total_steps:
                    checkpoint()
                if opt_step >= total_steps:
                    done = True
                    break
        # flush a partial accumulation group at epoch end so the tail of
        # the data still reaches the optimizer — and eval+save it too, or
        # the lowest-LR steps of the run never make it to the adapter
        if not done and micro % args.accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            checkpoint()
    if best < 0:  # no eval happened (tiny runs)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
    log.close()
    print(f"done. best dev acc={best:.4f} adapter={args.out} opt_steps={opt_step}")


if __name__ == "__main__":
    main()
