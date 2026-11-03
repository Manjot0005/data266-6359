"""HW4.a part 1: BERT sentiment fine-tune on a Twitter slice.

Records: row range, per-epoch train/val loss + val acc, epoch where val stopped improving and
what training past it cost, confusion matrix, per-class P/R, 5 misclassified examples from the
worst class, peak GPU memory and wall-clock training time (the HW4.5 baseline), save + reload +
custom inference.

Example (Kaggle/Colab):
  python hw4a_train.py --csv twitter.csv --text-col text --label-col sentiment --slice $SLICE --seed $SEED
  # Sentiment140 (no header, label col 0, text col 5):
  python hw4a_train.py --csv training.1600000.processed.noemoticon.csv --no-csv-header \
      --label-col 0 --text-col 5 --slice $SLICE --seed $SEED
"""
import os, time
import torch
from common import Cfg, build_datasets, train_and_eval, class_report, save_json, card_slug


def main():
    cfg, _ = Cfg.from_cli()
    cfg.save_model = cfg.save_model or "bert_twitter_hw4"
    dev = "cuda"
    splits, tok, names, rr, test_texts = build_datasets(cfg)
    print("Row range used:", rr)

    t0 = time.perf_counter()
    res = train_and_eval(cfg, splits, names, dev)
    wall = time.perf_counter() - t0                  # includes per-epoch validation
    model = res.pop("_model")
    preds, gold = res.pop("_test_preds"), res.pop("_test_gold")
    rep = class_report(gold, preds, names)

    w = names.index(rep["worst_class"])
    mis = [{"text": test_texts[i], "gold": names[g], "pred": names[p], "diagnosis": "TODO: one line"}
           for i, (g, p) in enumerate(zip(gold, preds)) if g == w and p != g][:5]

    # save, reload, custom inference
    model.save_pretrained(cfg.save_model); tok.save_pretrained(cfg.save_model)
    from transformers import BertForSequenceClassification, BertTokenizer
    m2 = BertForSequenceClassification.from_pretrained(cfg.save_model).to(dev).eval()
    t2 = BertTokenizer.from_pretrained(cfg.save_model)
    custom = ["I absolutely love this new phone!", "Worst customer service ever, never again.",
              "The flight leaves at 6pm tomorrow."]
    with torch.no_grad():
        enc = t2(custom, padding=True, truncation=True, max_length=cfg.max_len, return_tensors="pt").to(dev)
        pr = m2(**enc).logits.softmax(-1).cpu()
    inference = [{"text": s, "pred": names[int(p.argmax())], "probs": [round(float(x), 3) for x in p]}
                 for s, p in zip(custom, pr)]

    out = {**res, "row_range": rr, "wall_clock_incl_eval_s": wall, "class_report": rep,
           "worst_class_misclassified": mis, "custom_inference": inference}
    save_json(out, f"{cfg.out}/{card_slug()}/hw4a_train.json")
    print(f"\nRows {rr['first_row']}..{rr['last_row']} ({rr['n_rows']}) | best val epoch {res['best_val_epoch']} | "
          f"test acc {res['test_acc']:.4f} | peak mem {res['peak_mem_alloc_gb']:.2f} GB | train time {res['train_time_s']:.1f}s")
    print("Worst class:", rep["worst_class"], "-> fill in the 5 diagnoses in the JSON / report.")


if __name__ == "__main__":
    main()
