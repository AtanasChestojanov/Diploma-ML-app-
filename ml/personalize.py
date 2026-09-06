# on-device head personalization - export the trainable head

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core



import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np




def verify_encoder_tflite(tflite_bytes, seq_len, emb_dim):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    for det in interpreter.get_input_details():
        shape = list(det["shape"])
        resized = [1 if i == 0 else (int(d) if d and int(d) > 0 else seq_len)
                   for i, d in enumerate(shape)]
        interpreter.resize_tensor_input(det["index"], resized, strict=False)
    interpreter.allocate_tensors()
    for det in interpreter.get_input_details():
        interpreter.set_tensor(det["index"], np.zeros(det["shape"], dtype=det["dtype"]))
    interpreter.invoke()
    out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    return list(out.shape)


def export_b() -> None:
    parser = argparse.ArgumentParser(description="Phase B on-device export.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    args = parser.parse_args()

    config = core.load_config(args.config)
    core.apply_training_snapshot(config)
    core.set_global_seeds(config["seed"])
    core.ensure_output_dirs(config)

    encoder_type = config["model"].get("encoder_type", "cnn")
    models_dir = core.resolve_path(config["paths"]["models_dir"])
    exported_dir = core.resolve_path(config["paths"]["exported_dir"])
    exported_dir.mkdir(parents=True, exist_ok=True)
    od = config["ondevice"]
    seq_len = (config["model"]["mobilebert"]["seq_len"]
               if encoder_type == "mobilebert" else config["preprocessing"]["max_len"])

    print("=" * 70)
    print(f"Phase B on-device export (encoder_type={encoder_type})")
    print("=" * 70)

    full_model, encoder, head = core.build_model(config)
    encoder.load_weights(str(models_dir / "encoder.weights.h5"))
    head.load_weights(str(models_dir / "head.weights.h5"))
    emb_dim = int(head.input_shape[-1])
    print(f"[phaseB] Feature (embedding) dimension: {emb_dim}")

    report = {"encoder_type": encoder_type, "emb_dim": emb_dim}

    encoder_sm = models_dir / "encoder"
    enc_mode = od.get("encoder_variant", "dynamic_range")
    print(f"[phaseB] Converting frozen encoder ('{enc_mode}')...", flush=True)
    enc_bytes, enc_select = core.convert_for_deployment(encoder_sm, mode=enc_mode)
    enc_path = exported_dir / f"{encoder_type}_encoder.tflite"
    enc_path.write_bytes(enc_bytes)
    enc_out_shape = verify_encoder_tflite(enc_bytes, seq_len, emb_dim)
    report["encoder"] = {
        "file": enc_path.name,
        "variant": enc_mode,
        "size_mb": round(len(enc_bytes) / 1e6, 3),
        "select_tf_ops": enc_select,
        "output_shape": enc_out_shape,
    }
    print(f"[phaseB] encoder: {report['encoder']['size_mb']} MB | out {enc_out_shape} "
          f"-> {enc_path}")

    print("[phaseB] Building head-trainer (train/infer/save/restore)...", flush=True)
    saved_dir = models_dir / "head_trainer_savedmodel"
    head_bytes = core.build_head_trainer_tflite(
        head, emb_dim, od["learning_rate"], saved_dir
    )
    head_path = exported_dir / f"{encoder_type}_head_trainer.tflite"
    head_path.write_bytes(head_bytes)
    print(f"[phaseB] head_trainer: {len(head_bytes) / 1e6:.3f} MB -> {head_path}")

    print("[phaseB] Verifying head-trainer signatures...", flush=True)
    verify = core.verify_head_trainer(head_bytes, emb_dim)
    report["head_trainer"] = {
        "file": head_path.name,
        "size_mb": round(len(head_bytes) / 1e6, 3),
        "needs_select_tf_ops": True,
        **verify,
    }

    vec_dir = core.resolve_path(config["paths"]["vectorizer_dir"])
    tok_out = exported_dir / f"tokenizer_{encoder_type}"
    if vec_dir.is_dir() and not tok_out.exists():
        shutil.copytree(vec_dir, tok_out)
        print(f"[phaseB] Copied tokenizer artifacts -> {tok_out}")

    report_path = exported_dir / f"export_report_phaseB_{encoder_type}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("-" * 70)
    print(json.dumps(report, indent=2))
    print(f"[phaseB] Wrote report -> {report_path}")
    tl = report["head_trainer"]
    if tl.get("train_loss_decreased"):
        print(f"[phaseB] OK: on-device training reduces loss "
              f"({tl['train_loss_first']} -> {tl['train_loss_last']}).")
    else:
        print("[phaseB] WARNING: train loss did not decrease — review above.")



import argparse
import sys
from pathlib import Path




def export_head() -> None:
    parser = argparse.ArgumentParser(description="Re-export ONLY the head trainer.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--lr", type=float, default=None,
                        help="override the learning_rate baked into the trainer "
                             "(does not modify config.yaml)")
    args = parser.parse_args()

    config = core.load_config(args.config)
    core.apply_training_snapshot(config)
    core.set_global_seeds(config["seed"])
    core.ensure_output_dirs(config)

    encoder_type = config["model"].get("encoder_type", "cnn")
    models_dir = core.resolve_path(config["paths"]["models_dir"])
    exported_dir = core.resolve_path(config["paths"]["exported_dir"])
    exported_dir.mkdir(parents=True, exist_ok=True)
    od = config["ondevice"]

    print("=" * 70)
    print(f"Head-trainer re-export only (encoder_type={encoder_type})")
    print("=" * 70)

    _full, _encoder, head = core.build_model(config)
    head.load_weights(str(models_dir / "head.weights.h5"))
    emb_dim = int(head.input_shape[-1])
    lr = args.lr if args.lr is not None else od["learning_rate"]
    src = "override" if args.lr is not None else "config"
    print(f"[head-only] embedding dim: {emb_dim} | learning_rate: {lr} ({src})")

    saved_dir = models_dir / "head_trainer_savedmodel"
    head_bytes = core.build_head_trainer_tflite(
        head, emb_dim, lr, saved_dir
    )
    head_path = exported_dir / f"{encoder_type}_head_trainer.tflite"
    head_path.write_bytes(head_bytes)
    print(f"[head-only] wrote {len(head_bytes) / 1e6:.3f} MB -> {head_path}")

    verify = core.verify_head_trainer(head_bytes, emb_dim)
    print(f"[head-only] signatures: {verify.get('signatures')}")
    print(f"[head-only] train loss {verify.get('train_loss_first')} -> "
          f"{verify.get('train_loss_last')} (decreased: {verify.get('train_loss_decreased')}); "
          f"save/restore: {verify.get('save_restore')}")
    print("\nNext: copy this .tflite over "
          "App/app/src/main/assets/mobilebert_head_trainer.tflite, then rebuild the app.")



import argparse
import sys
from pathlib import Path

import numpy as np


import tensorflow as tf  # noqa: E402



def reexport() -> None:
    ap = argparse.ArgumentParser(description="Re-export head trainer from an existing tflite.")
    ap.add_argument("--source", required=True,
                    help="existing head_trainer .tflite to pull trained weights from")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"source tflite not found: {src}")

    config = core.load_config(args.config)
    core.apply_training_snapshot(config)
    core.set_global_seeds(config["seed"])
    core.ensure_output_dirs(config)

    encoder_type = config["model"].get("encoder_type", "cnn")
    models_dir = core.resolve_path(config["paths"]["models_dir"])
    exported_dir = core.resolve_path(config["paths"]["exported_dir"])
    exported_dir.mkdir(parents=True, exist_ok=True)
    od = config["ondevice"]

    print("=" * 70)
    print("Re-export head trainer from existing tflite (dropout-off, deterministic)")
    print(f"source weights: {src}")
    print("=" * 70)

    interp = tf.lite.Interpreter(model_path=str(src))
    interp.allocate_tensors()
    save_run = interp.get_signature_runner("save")
    flat = np.asarray(save_run(unused=np.zeros([1], np.float32))["weights"]).reshape(-1)
    infer_src = interp.get_signature_runner("infer")
    print(f"[reexport] extracted {flat.shape[0]} head weights from source")

    _full, _enc, head = core.build_model(config)
    emb_dim = int(head.input_shape[-1])
    offset = 0
    for v in head.trainable_variables:
        n = int(np.prod(v.shape.as_list()))
        v.assign(flat[offset:offset + n].reshape(v.shape.as_list()))
        offset += n
    if offset != flat.shape[0]:
        raise SystemExit(f"weight size mismatch: assigned {offset}, have {flat.shape[0]}")
    print(f"[reexport] loaded weights into fresh head ({offset} floats, emb_dim={emb_dim})")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, emb_dim)).astype(np.float32)
    src_probs = np.asarray(infer_src(x=x)["probabilities"])
    out = head(x, training=False).numpy()
    looks_like_probs = bool(np.all(out >= -1e-6) and np.allclose(out.sum(axis=1), 1.0, atol=1e-3))
    new_probs = out if looks_like_probs else tf.nn.softmax(out, axis=-1).numpy()
    max_diff = float(np.max(np.abs(src_probs - new_probs)))
    print(f"[reexport] infer parity check: max|delta| = {max_diff:.2e}")
    if max_diff > 1e-3:
        raise SystemExit(
            f"ABORT: rebuilt head does not match source (max diff {max_diff}). "
            "Weight ordering mismatch — do NOT deploy this model."
        )
    print("[reexport] parity OK — weights loaded correctly.")

    head.save_weights(str(models_dir / "head.weights.h5"))
    print(f"[reexport] wrote {models_dir / 'head.weights.h5'} (for future exports)")

    saved_dir = models_dir / "head_trainer_savedmodel"
    head_bytes = core.build_head_trainer_tflite(head, emb_dim, od["learning_rate"], saved_dir)
    out_path = exported_dir / f"{encoder_type}_head_trainer.tflite"
    out_path.write_bytes(head_bytes)
    print(f"[reexport] wrote {len(head_bytes) / 1e6:.3f} MB -> {out_path}")

    verify = core.verify_head_trainer(head_bytes, emb_dim)
    print(f"[reexport] signatures: {verify.get('signatures')}")
    print(f"[reexport] train loss {verify.get('train_loss_first')} -> "
          f"{verify.get('train_loss_last')} (decreased: {verify.get('train_loss_decreased')}); "
          f"save/restore: {verify.get('save_restore')}")
    print("\nNext: copy this .tflite over "
          "App/app/src/main/assets/mobilebert_head_trainer.tflite, then rebuild the app.")


def _main():
    stages = {"export_b": export_b, "export_head": export_head, "reexport": reexport}
    if len(sys.argv) < 2 or sys.argv[1] not in stages:
        sys.exit("usage: python ml/personalize.py <stage>  [" + " | ".join(stages) + "]")
    stages[sys.argv.pop(1)]()


if __name__ == "__main__":
    _main()
