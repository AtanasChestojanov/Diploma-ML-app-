# base model training

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core



import argparse
import json
import sys
from pathlib import Path




def prepare() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Dynahate dataset.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (defaults to ml/config/config.yaml).",
    )
    args = parser.parse_args()
    config = core.load_config(args.config)
    core.set_global_seeds(config["seed"])
    core.ensure_output_dirs(config)

    core.download_raw_csv(config)
    df = core.load_raw_dataframe(config)
    df = core._clean_text(df, config)
    df = core._normalize_labels(df, config)
    splits = core.split_dataframe(df, config)
    summary = core.summarize_splits(splits, config)
    paths = core.save_splits(splits, config)

    summary_path = (
        core.resolve_path(config["paths"]["processed_data_dir"]) / "data_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 70)
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print(f"[data] Wrote summary -> {summary_path}")
    print("-" * 70)
    print("Done. Processed splits:")
    for name, p in paths.items():
        print(f"  {name:11s} -> {p}")



import argparse
import sys
from pathlib import Path




def vectorizer() -> None:
    parser = argparse.ArgumentParser(description="Build/export tokenizer artifacts.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=15,
        help="How many real validation texts to include in the examples file.",
    )
    args = parser.parse_args()

    config = core.load_config(args.config)
    core.set_global_seeds(config["seed"])
    core.ensure_output_dirs(config)
    encoder_type = config["model"].get("encoder_type", "cnn")
    print("=" * 70)
    print(f"Building tokenizer artifacts (encoder_type={encoder_type})")
    print("=" * 70)

    core.export_encoder_artifacts(config, num_examples=args.num_examples)

    print("-" * 70)
    print("Done. Artifacts written to:")
    print(f"  {core.resolve_path(config['paths']['vectorizer_dir'])}")



import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml




def compute_class_weights(labels: np.ndarray) -> dict:
    """Inverse-frequency class weights for a binary label array."""
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    total = counts.sum()
    return {i: float(total / (2.0 * counts[i])) for i in range(2)}


def train() -> None:
    import tensorflow as tf
    from tensorflow import keras

    parser = argparse.ArgumentParser(description="Train the hate-speech classifier.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Build the model, run ONE forward pass on a single batch, print "
        "initial loss + logit/encoder magnitudes, then exit (no training).",
    )
    args = parser.parse_args()

    config = core.load_config(args.config)
    core.set_global_seeds(config["seed"])
    core.ensure_output_dirs(config)

    models_dir = core.resolve_path(config["paths"]["models_dir"])
    metrics_dir = core.resolve_path(config["paths"]["metrics_dir"])
    tr = config["training"]
    encoder_type = config["model"].get("encoder_type", "cnn")

    if encoder_type == "mobilebert":
        mb = config["model"]["mobilebert"]
        if mb.get("fine_tune_encoder", True):
            learning_rate, epochs = mb["encoder_lr"], mb["epochs"]
            print("[train] MobileBERT: fine-tuning the full encoder.")
        else:
            learning_rate, epochs = mb["head_lr"], mb["head_epochs"]
            print("[train] MobileBERT: encoder FROZEN, training head only.")
        smoke_seq_len = mb["seq_len"]
    else:
        learning_rate, epochs = tr["learning_rate"], tr["epochs"]
        smoke_seq_len = config["preprocessing"]["max_len"]

    print("=" * 70)
    print(f"Training hate-speech classifier (encoder_type={encoder_type})")
    print("=" * 70)

    datasets = core.build_datasets(config)
    _, train_labels = core.load_split(config, "train")

    class_weight = None
    if tr.get("use_class_weights", False):
        class_weight = compute_class_weights(train_labels)
        print(f"[train] Class weights: {class_weight}")

    full_model, encoder, head = core.build_model(config)
    use_logits = core.output_is_logits(config)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=use_logits)
    fine_tuning = encoder_type == "mobilebert" and config["model"]["mobilebert"].get(
        "fine_tune_encoder", True
    )
    if fine_tuning:
        from transformers import WarmUp

        steps_per_epoch = math.ceil(len(train_labels) / tr["batch_size"])
        total_steps = steps_per_epoch * epochs
        warmup_steps = max(1, int(0.1 * total_steps))
        decay = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=max(1, total_steps - warmup_steps),
            end_learning_rate=learning_rate * 0.1,
            power=1.0,
        )
        schedule = WarmUp(
            initial_learning_rate=learning_rate,
            decay_schedule_fn=decay,
            warmup_steps=warmup_steps,
        )
        weight_decay = float(mb.get("weight_decay", 0.0))
        if weight_decay > 0.0:
            optimizer = keras.optimizers.AdamW(
                learning_rate=schedule, weight_decay=weight_decay
            )
            opt_name = f"AdamW(wd={weight_decay})"
        else:
            optimizer = keras.optimizers.Adam(learning_rate=schedule)
            opt_name = "Adam"
        print(
            f"[train] Optimizer: {opt_name} + WarmUp({warmup_steps}/{total_steps}), "
            "loss from_logits=True"
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    full_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    full_model.summary(print_fn=lambda s: print("[model] " + s))

    if args.diagnose:
        print("[diagnose] Running one-batch forward pass (no training)...")
        for batch_inputs, batch_labels in datasets["train"].take(1):
            enc_out = encoder(batch_inputs, training=True).numpy()
            preds = full_model(batch_inputs, training=True)
            batch_loss = float(tf.reduce_mean(loss(batch_labels, preds)))
            preds = preds.numpy()
            kind = "logits" if use_logits else "probs"
            print(f"[diagnose] encoder output: min={enc_out.min():.3f} "
                  f"max={enc_out.max():.3f} mean={enc_out.mean():.3f}")
            enc_batch_std = float(enc_out.std(axis=0).mean())
            print(f"[diagnose] encoder per-feature std ACROSS batch = "
                  f"{enc_batch_std:.4f}  (~0 => identical/degenerate features)")
            print(f"[diagnose] head output ({kind}): min={preds.min():.3f} "
                  f"max={preds.max():.3f} mean={preds.mean():.3f}")
            print(f"[diagnose] initial batch loss = {batch_loss:.4f} "
                  "(expect ~0.69 for 2 balanced classes)")

            if encoder_type == "mobilebert":
                from transformers import TFMobileBertModel

                raw = TFMobileBertModel.from_pretrained(
                    config["model"]["mobilebert"]["hf_model_name"]
                )
                ro = raw(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    token_type_ids=batch_inputs["token_type_ids"],
                )
                lh = ro.last_hidden_state
                m = tf.cast(batch_inputs["attention_mask"], lh.dtype)[..., None]
                pooled = (tf.reduce_sum(lh * m, axis=1)
                          / tf.maximum(tf.reduce_sum(m, axis=1), 1e-9)).numpy()
                print(f"[diagnose] BARE HF (eager) mean-pooled per-feature std "
                      f"ACROSS batch = {pooled.std(axis=0).mean():.4f}")
                print(f"[diagnose] BARE HF last_hidden_state range: "
                      f"min={float(tf.reduce_min(lh)):.1f} "
                      f"max={float(tf.reduce_max(lh)):.1f}")
        print("[diagnose] Done — exiting before training.")
        return

    best_ckpt = models_dir / "best.weights.h5"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=tr["early_stopping_patience"],
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_ckpt),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.CSVLogger(str(metrics_dir / "training_log.csv")),
    ]

    history = full_model.fit(
        datasets["train"],
        validation_data=datasets["validation"],
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    if best_ckpt.exists():
        full_model.load_weights(str(best_ckpt))
        print(f"[train] Reloaded best-epoch weights from {best_ckpt.name}")

    val_loss, val_acc = full_model.evaluate(datasets["validation"], verbose=0)
    print(f"[train] Best-epoch validation: loss={val_loss:.4f} acc={val_acc:.4f}")

    saved_model_dir = models_dir / "saved_model"
    encoder_dir = models_dir / "encoder"
    head_dir = models_dir / "head"

    full_model.save(str(saved_model_dir), save_format="tf")
    encoder.save(str(encoder_dir), save_format="tf")
    head.save(str(head_dir), save_format="tf")

    if encoder_type == "cnn":
        full_model.save(str(models_dir / "full_model.keras"))
    else:
        print("[train] Skipping full_model.keras for non-cnn encoder.")

    encoder.save_weights(str(models_dir / "encoder.weights.h5"))
    head.save_weights(str(models_dir / "head.weights.h5"))

    history_json = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(models_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history_json, f, indent=2)

    with open(models_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"[train] Saved full SavedModel  -> {saved_model_dir}")
    print(f"[train] Saved encoder          -> {encoder_dir} (+ encoder.weights.h5)")
    print(f"[train] Saved head             -> {head_dir} (+ head.weights.h5)")
    print(f"[train] Saved history.json, config_snapshot.yaml")

    core.plot_training_history(history.history, metrics_dir)

    print("-" * 70)
    print("TFLite conversion smoke test (default float conversion):")
    smoke_dir = models_dir / "smoke"
    full_ok = core.smoke_test_saved_model(
        saved_model_dir, smoke_dir / "full_model_smoke.tflite", "full_model",
        seq_len=smoke_seq_len,
    )
    enc_ok = core.smoke_test_saved_model(
        encoder_dir, smoke_dir / "encoder_smoke.tflite", "encoder",
        seq_len=smoke_seq_len,
    )
    print("-" * 70)
    if full_ok and enc_ok:
        print("Done. Both models converted and ran in the TFLite interpreter.")
    else:
        print(
            "Done, but at least one conversion FAILED — review the [smoke] lines "
            "above before building Step 5/6."
        )



import argparse
import json
import sys
from pathlib import Path

import numpy as np




def class_names_from_config(config) -> list:
    """Class names ordered by integer id (e.g. ['nothate', 'hate'])."""
    label_map = config["dataset"]["label_map"]
    return [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]


def evaluate() -> None:
    import tensorflow as tf
    from tensorflow import keras

    parser = argparse.ArgumentParser(description="Evaluate on the test split.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--split", default="test", help="Split to evaluate.")
    parser.add_argument(
        "--tune-split",
        default="validation",
        help="Split to tune the decision threshold on (then applied to --split). "
        "Set to '' or the same as --split to disable rigorous tuning.",
    )
    args = parser.parse_args()

    config = core.load_config(args.config)
    core.apply_training_snapshot(config)
    core.set_global_seeds(config["seed"])

    models_dir = core.resolve_path(config["paths"]["models_dir"])
    metrics_dir = core.resolve_path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    class_names = class_names_from_config(config)

    encoder_weights = models_dir / "encoder.weights.h5"
    head_weights = models_dir / "head.weights.h5"
    if not encoder_weights.is_file() or not head_weights.is_file():
        raise FileNotFoundError(
            f"Missing weights at {encoder_weights} / {head_weights}. Train first."
        )
    print(f"[eval] Rebuilding model and loading weights from {models_dir}")
    model, encoder, head = core.build_model(config)
    encoder.load_weights(str(encoder_weights))
    head.load_weights(str(head_weights))

    texts, y_true = core.load_split(config, args.split)
    inputs = core.texts_to_inputs(config, texts)

    print(f"[eval] Predicting on {len(texts):,} '{args.split}' examples...")
    preds = model.predict(inputs, batch_size=config["training"]["batch_size"], verbose=1)
    preds = np.asarray(preds)
    if core.output_is_logits(config):
        preds = tf.nn.softmax(preds, axis=1).numpy()
    probs = preds
    y_pred = probs.argmax(axis=1)
    y_score = probs[:, 1]

    results = core.compute_classification_metrics(
        y_true, y_pred, y_score, class_names
    )
    results["split"] = args.split

    tune_split = args.tune_split
    if tune_split and tune_split != args.split:
        print(f"[eval] Tuning decision threshold on '{tune_split}'...")
        tune_texts, tune_y = core.load_split(config, tune_split)
        tune_inputs = core.texts_to_inputs(config, tune_texts)
        tune_preds = np.asarray(
            model.predict(tune_inputs, batch_size=config["training"]["batch_size"], verbose=0)
        )
        if core.output_is_logits(config):
            tune_preds = tf.nn.softmax(tune_preds, axis=1).numpy()
        tune_score = tune_preds[:, 1]

        best_acc = core.best_threshold(tune_y, tune_score, "accuracy")
        best_f1 = core.best_threshold(tune_y, tune_score, "f1_hate")
        results["threshold_tuning"] = {
            "tuned_on": tune_split,
            "default_0.5": core.metrics_at_threshold(y_true, y_score, 0.5),
            "best_accuracy": {
                "threshold": best_acc["threshold"],
                f"{tune_split}_accuracy": best_acc["accuracy"],
                f"{args.split}_at_threshold": core.metrics_at_threshold(
                    y_true, y_score, best_acc["threshold"]
                ),
            },
            "best_f1": {
                "threshold": best_f1["threshold"],
                f"{tune_split}_f1_hate": best_f1["f1_hate"],
                f"{args.split}_at_threshold": core.metrics_at_threshold(
                    y_true, y_score, best_f1["threshold"]
                ),
            },
        }

    out_json = metrics_dir / f"{args.split}_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Wrote metrics -> {out_json}")

    core.plot_confusion_matrix(
        results["confusion_matrix"], class_names,
        metrics_dir / "confusion_matrix.png",
    )
    core.plot_confusion_matrix(
        results["confusion_matrix"], class_names,
        metrics_dir / "confusion_matrix_normalized.png", normalize=True,
    )
    core.plot_roc_curve(y_true, y_score, metrics_dir / "roc_curve.png")
    core.plot_precision_recall_curve(y_true, y_score, metrics_dir / "pr_curve.png")

    # Readable summary.
    print("-" * 70)
    print(f"{args.split.upper()} results ({results['num_examples']:,} examples)")
    print(f"  accuracy : {results['accuracy']:.4f}")
    print(f"  roc_auc  : {results['roc_auc']:.4f}")
    for name in class_names:
        pc = results["per_class"][name]
        print(
            f"  {name:8s} precision={pc['precision']:.4f} "
            f"recall={pc['recall']:.4f} f1={pc['f1-score']:.4f} "
            f"support={int(pc['support'])}"
        )
    ma = results["macro_avg"]
    print(
        f"  macro    precision={ma['precision']:.4f} "
        f"recall={ma['recall']:.4f} f1={ma['f1-score']:.4f}"
    )
    print(f"  confusion_matrix (rows=true {class_names}): "
          f"{results['confusion_matrix']}")
    if "threshold_tuning" in results:
        tt = results["threshold_tuning"]
        d = tt["default_0.5"]
        ba = tt["best_accuracy"]
        a_at = ba[f"{args.split}_at_threshold"]
        print(f"  threshold tuning (tuned on '{tt['tuned_on']}'):")
        print(f"    default  thr=0.50 -> acc={d['accuracy']:.4f} "
              f"f1_hate={d['f1_hate']:.4f}")
        print(f"    best-acc thr={ba['threshold']:.2f} -> "
              f"{args.split} acc={a_at['accuracy']:.4f} f1_hate={a_at['f1_hate']:.4f}")
    print("-" * 70)



import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np




def build_inference_model(config, models_dir):
    """Rebuild the model, load weights, and ensure a probability output."""
    from tensorflow import keras

    encoder_weights = models_dir / "encoder.weights.h5"
    head_weights = models_dir / "head.weights.h5"
    if not encoder_weights.is_file() or not head_weights.is_file():
        raise FileNotFoundError(
            f"Missing weights at {encoder_weights} / {head_weights}. Train first."
        )

    full_model, encoder, head = core.build_model(config)
    encoder.load_weights(str(encoder_weights))
    head.load_weights(str(head_weights))

    if core.output_is_logits(config):
        probs = keras.layers.Softmax(name="probabilities")(full_model.output)
        inference_model = keras.Model(full_model.inputs, probs, name="inference_model")
    else:
        inference_model = full_model
    return inference_model


def make_representative_dataset(config, input_names, num_samples):
    """Yield calibration samples (batch size 1) for int8 quantization."""
    texts, _ = core.load_split(config, "train")
    texts = texts[:num_samples]
    inputs = core.texts_to_inputs(config, texts)

    def gen():
        for i in range(len(texts)):
            if isinstance(inputs, dict):
                yield [inputs[name][i : i + 1] for name in input_names]
            else:
                yield [inputs[i : i + 1]]

    return gen


def export_a() -> None:
    parser = argparse.ArgumentParser(description="Phase A TFLite export.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument(
        "--int8", action="store_true", help="Also attempt full-integer (int8) export."
    )
    args = parser.parse_args()

    config = core.load_config(args.config)
    core.apply_training_snapshot(config)
    core.set_global_seeds(config["seed"])
    core.ensure_output_dirs(config)

    encoder_type = config["model"].get("encoder_type", "cnn")
    models_dir = core.resolve_path(config["paths"]["models_dir"])
    exported_dir = core.resolve_path(config["paths"]["exported_dir"])
    exported_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Phase A export (encoder_type={encoder_type})")
    print("=" * 70)

    inference_model = build_inference_model(config, models_dir)
    inf_dir = models_dir / "inference_savedmodel"
    inference_model.save(str(inf_dir), save_format="tf")
    print(f"[export] Wrote inference SavedModel -> {inf_dir}")

    variants = list(config["export"]["variants"])
    if args.int8 and "int8" not in variants:
        variants.append("int8")

    representative_dataset = None
    if "int8" in variants:
        representative_dataset = make_representative_dataset(
            config, inference_model.input_names,
            config["export"]["representative_samples"],
        )

    report = {"encoder_type": encoder_type, "variants": {}}
    for mode in variants:
        out_path = exported_dir / f"{encoder_type}_{mode}.tflite"
        print(f"[export] Converting variant '{mode}'...", flush=True)
        try:
            rep = representative_dataset if mode == "int8" else None
            tflite_model, used_select = core.convert_for_deployment(
                inf_dir, mode=mode, representative_dataset=rep
            )
            out_path.write_bytes(tflite_model)
            size_mb = len(tflite_model) / 1e6
            report["variants"][mode] = {
                "file": out_path.name,
                "size_mb": round(size_mb, 3),
                "select_tf_ops": used_select,
            }
            flex = " (needs SELECT_TF_OPS)" if used_select else ""
            print(f"[export]   {mode}: {size_mb:.2f} MB{flex} -> {out_path}")
        except Exception as exc:  # noqa: BLE001
            report["variants"][mode] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[export]   {mode}: FAILED -> {type(exc).__name__}: {exc}")

    vec_dir = core.resolve_path(config["paths"]["vectorizer_dir"])
    tok_out = exported_dir / f"tokenizer_{encoder_type}"
    if vec_dir.is_dir():
        if tok_out.exists():
            shutil.rmtree(tok_out)
        shutil.copytree(vec_dir, tok_out)
        print(f"[export] Copied tokenizer artifacts -> {tok_out}")

    report_path = exported_dir / f"export_report_{encoder_type}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("-" * 70)
    print(json.dumps(report, indent=2))
    print(f"[export] Wrote report -> {report_path}")



import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np




def _seq_len_for(config) -> int:
    if config["model"].get("encoder_type", "cnn") == "mobilebert":
        return config["model"]["mobilebert"]["seq_len"]
    return config["preprocessing"]["max_len"]


def benchmark_one(tf, path, seq_len, num_runs, warmup_runs, num_threads):
    size_mb = path.stat().st_size / 1e6
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=num_threads if num_threads else None,
    )

    for det in interpreter.get_input_details():
        shape = list(det["shape"])
        resized = [
            1 if i == 0 else (int(d) if d and int(d) > 0 else seq_len)
            for i, d in enumerate(shape)
        ]
        interpreter.resize_tensor_input(det["index"], resized, strict=False)
    interpreter.allocate_tensors()

    for det in interpreter.get_input_details():
        interpreter.set_tensor(det["index"], np.zeros(det["shape"], dtype=det["dtype"]))

    for _ in range(warmup_runs):
        interpreter.invoke()

    times_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.invoke()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    stats = {
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }
    return round(size_mb, 3), stats


def benchmark() -> None:
    import tensorflow as tf

    parser = argparse.ArgumentParser(description="Benchmark exported TFLite models.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument(
        "--threads", type=int, default=0, help="Interpreter threads (0 = default)."
    )
    args = parser.parse_args()

    config = core.load_config(args.config)
    core.apply_training_snapshot(config)  # match the trained model's seq_len
    exported_dir = core.resolve_path(config["paths"]["exported_dir"])
    seq_len = _seq_len_for(config)
    bench = config["benchmark"]
    num_runs = bench["num_runs"]
    warmup_runs = bench["warmup_runs"]

    tflite_files = sorted(exported_dir.glob("*.tflite"))
    if not tflite_files:
        raise FileNotFoundError(
            f"No .tflite files in {exported_dir}. Run 05_export_phase_a.py first."
        )

    print("=" * 70)
    print(f"Benchmarking {len(tflite_files)} model(s) | runs={num_runs} "
          f"warmup={warmup_runs} threads={args.threads or 'default'} seq_len={seq_len}")
    print("=" * 70)

    results = {}
    for path in tflite_files:
        size_mb, stats = benchmark_one(
            tf, path, seq_len, num_runs, warmup_runs, args.threads
        )
        results[path.name] = {"size_mb": size_mb, "latency": stats}
        print(f"  {path.name:34s} {size_mb:8.2f} MB | "
              f"mean {stats['mean_ms']:8.2f} ms | p95 {stats['p95_ms']:8.2f} ms")

    out_path = exported_dir / "benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "num_runs": num_runs,
                    "warmup_runs": warmup_runs,
                    "threads": args.threads or "default",
                    "seq_len": seq_len,
                    "device": "workstation CPU",
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print("-" * 70)
    print(f"[bench] Wrote -> {out_path}")


def _main():
    stages = {"prepare": prepare, "vectorizer": vectorizer, "train": train, "evaluate": evaluate, "export_a": export_a, "benchmark": benchmark}
    if len(sys.argv) < 2 or sys.argv[1] not in stages:
        sys.exit("usage: python ml/train_base.py <stage>  [" + " | ".join(stages) + "]")
    stages[sys.argv.pop(1)]()


if __name__ == "__main__":
    _main()
