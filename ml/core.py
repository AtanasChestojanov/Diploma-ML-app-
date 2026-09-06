# data prep, MobileBERT encoder + head, TFLite conversion, metrics, and the on-device head trainer

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "ml" / "config" / "config.yaml"

def load_config(config_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    required = ["seed", "dataset", "preprocessing", "model", "training", "paths"]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Config is missing required section(s): {missing}")


def resolve_path(relative_path: str | os.PathLike) -> Path:
    p = Path(relative_path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    for rel in config["paths"].values():
        resolve_path(rel).mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def apply_training_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = resolve_path(config["paths"]["models_dir"]) / "config_snapshot.yaml"
    if not snapshot.is_file():
        return config
    with open(snapshot, "r", encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    for section in ("model", "preprocessing"):
        if section in saved:
            config[section] = saved[section]
    print(
        f"[config] Matched trained model from {snapshot.name} "
        f"(encoder_type={config['model'].get('encoder_type')})"
    )
    return config

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


VOCAB_FILENAME = "vocab.txt"
VOCAB_JSON_FILENAME = "vocab.json"
VECTORIZER_CONFIG_FILENAME = "vectorizer_config.json"
TOKENIZATION_EXAMPLES_FILENAME = "tokenization_examples.json"

PAD_INDEX = 0
OOV_INDEX = 1

def load_split(config: Dict[str, Any], split: str) -> Tuple[List[str], np.ndarray]:
    """Load a cleaned split CSV (produced by Step 2) as (texts, labels)."""
    text_col = config["dataset"]["text_column"]
    label_col = config["dataset"]["label_column"]
    csv_path = resolve_path(config["paths"]["processed_data_dir"]) / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Processed split not found: {csv_path}. Run "
            "ml/scripts/01_prepare_data.py first."
        )
    df = pd.read_csv(csv_path, encoding="utf-8")
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(np.int32).to_numpy()
    return texts, labels

def build_vectorizer(config: Dict[str, Any]) -> tf.keras.layers.TextVectorization:
    """Create an (unadapted) TextVectorization layer from config."""
    pp = config["preprocessing"]
    standardize = "lower" if get("lowercase", True) else None
    return tf.keras.layers.TextVectorization(
        max_tokens=pp["vocab_size"],
        standardize=standardize,
        split="whitespace",
        output_mode="int",
        output_sequence_length=pp["max_len"],
    )


def adapt_vectorizer(
    vectorizer: tf.keras.layers.TextVectorization, texts: List[str]
) -> None:
    """Learn the vocabulary from TRAIN texts only."""
    vectorizer.adapt(tf.constant(texts))


def save_vectorizer(
    vectorizer: tf.keras.layers.TextVectorization, config: Dict[str, Any]
) -> Dict[str, Path]:
    out_dir = resolve_path(config["paths"]["vectorizer_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = vectorizer.get_vocabulary(include_special_tokens=True)
    vocab_path = out_dir / VOCAB_FILENAME
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")

    vocab_json_path = out_dir / VOCAB_JSON_FILENAME
    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    pp = config["preprocessing"]
    cfg_path = out_dir / VECTORIZER_CONFIG_FILENAME
    vec_config = {
        "standardize": "lower" if get("lowercase", True) else None,
        "split": "whitespace",
        "output_mode": "int",
        "max_len": pp["max_len"],
        "max_tokens": pp["vocab_size"],
        "actual_vocab_size": len(vocab),
        "pad_index": PAD_INDEX,
        "oov_index": OOV_INDEX,
        "lowercase": bool(get("lowercase", True)),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vec_config, f, indent=2)

    print(f"[prep] Wrote vocabulary ({len(vocab):,} tokens) -> {vocab_path}")
    print(f"[prep] Wrote vectorizer config -> {cfg_path}")
    return {"vocab": vocab_path, "config": cfg_path}


def load_vectorizer(config: Dict[str, Any]) -> tf.keras.layers.TextVectorization:
    out_dir = resolve_path(config["paths"]["vectorizer_dir"])
    json_path = out_dir / VOCAB_JSON_FILENAME
    txt_path = out_dir / VOCAB_FILENAME

    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            tokens = json.load(f)
    elif txt_path.is_file():
        with open(txt_path, "r", encoding="utf-8") as f:
            tokens = f.read().split("\n")
        if tokens and tokens[-1] == "":  # drop trailing-newline artifact
            tokens = tokens[:-1]
    else:
        raise FileNotFoundError(
            f"Vocabulary not found: {json_path} or {txt_path}. Run "
            "ml/scripts/02_build_vectorizer.py first."
        )

    if tokens[:2] == ["", "[UNK]"]:
        vocab_terms = tokens[2:]
    else:
        vocab_terms = tokens

    vectorizer = build_vectorizer(config)
    vectorizer.set_vocabulary(vocab_terms)
    return vectorizer


def build_tf_datasets(
    config: Dict[str, Any],
    vectorizer: tf.keras.layers.TextVectorization,
) -> Dict[str, tf.data.Dataset]:
    batch_size = config["training"]["batch_size"]
    seed = config["seed"]
    autotune = tf.data.AUTOTUNE

    datasets: Dict[str, tf.data.Dataset] = {}
    for split in ("train", "validation", "test"):
        texts, labels = load_split(config, split)
        ids = vectorizer(tf.constant(texts)).numpy().astype(np.int32)
        ds = tf.data.Dataset.from_tensor_slices((ids, labels))
        if split == "train":
            ds = ds.shuffle(
                buffer_size=len(labels),
                seed=seed,
                reshuffle_each_iteration=True,
            )
        ds = ds.batch(batch_size).prefetch(autotune)
        datasets[split] = ds
    return datasets

def export_tokenization_examples(
    vectorizer: tf.keras.layers.TextVectorization,
    config: Dict[str, Any],
    sample_texts: List[str],
) -> Path:
    pp = config["preprocessing"]
    max_len = pp["max_len"]
    lowercase = bool(get("lowercase", True))

    ids_batch = vectorizer(tf.constant(sample_texts)).numpy().astype(int)

    examples = []
    for text, ids in zip(sample_texts, ids_batch):
        normalized = text.lower() if lowercase else text
        tokens = normalized.split()  # mirrors split="whitespace"
        token_ids = ids.tolist()
        non_pad = int(np.count_nonzero(ids != PAD_INDEX))
        examples.append(
            {
                "text": text,
                "tokens": tokens,
                "token_ids": token_ids,
                "non_pad_count": non_pad,
            }
        )

    payload = {
        "contract": {
            "standardize": "lower" if lowercase else None,
            "split": "whitespace",
            "max_len": max_len,
            "pad_index": PAD_INDEX,
            "oov_index": OOV_INDEX,
            "note": (
                "token_ids has length max_len: sequences are right-padded with "
                "pad_index and truncated to max_len. Unknown tokens map to "
                "oov_index. vocab.txt line number (0-based) == token ID."
            ),
        },
        "examples": examples,
    }

    out_dir = resolve_path(config["paths"]["vectorizer_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / TOKENIZATION_EXAMPLES_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[prep] Wrote {len(examples)} tokenization examples -> {out_path}")
    return out_path


def compute_oov_rate(
    vectorizer: tf.keras.layers.TextVectorization, texts: List[str]
) -> float:
    """Fraction of non-pad token positions that map to OOV (on given texts)."""
    ids = vectorizer(tf.constant(texts)).numpy()
    non_pad = ids != PAD_INDEX
    total = int(non_pad.sum())
    if total == 0:
        return 0.0
    oov = int(((ids == OOV_INDEX) & non_pad).sum())
    return oov / total

import hashlib
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def download_raw_csv(config: Dict[str, Any]) -> Path:
    ds = config["dataset"]
    dest = resolve_path(ds["raw_csv_path"])

    if dest.is_file():
        print(f"[data] Raw CSV already present: {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = ds["download_url"]
        print(f"[data] Downloading Dynahate v{ds['version']} CSV...")
        print(f"[data]   from: {url}")
        print(f"[data]   to:   {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"[data] Download complete ({dest.stat().st_size:,} bytes).")

    if ds.get("verify_checksum", False):
        _verify_checksum(dest, ds.get("expected_sha256"))

    return dest


def _verify_checksum(path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        print("[data] No expected checksum configured; skipping verification.")
        return

    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    actual = sha.hexdigest()

    if actual != expected_sha256:
        raise ValueError(
            "Checksum mismatch for raw CSV!\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            "The upstream file may have changed. Verify the source, then either "
            "update `dataset.expected_sha256` in config.yaml or set "
            "`dataset.verify_checksum: false`."
        )
    print(f"[data] Checksum OK (sha256={actual}).")


def load_raw_dataframe(config: Dict[str, Any]) -> pd.DataFrame:
    csv_path = resolve_path(config["dataset"]["raw_csv_path"])
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Raw CSV not found at {csv_path}. Run download_raw_csv() first "
            "(the prepare script does this automatically)."
        )

    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"[data] Loaded raw CSV: {len(df):,} rows, {df.shape[1]} columns.")

    ds = config["dataset"]
    required = [ds["text_column"], ds["label_column"], ds["split_column"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Raw CSV is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def _normalize_labels(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    label_col = config["dataset"]["label_column"]
    label_map = config["dataset"]["label_map"]

    raw = df[label_col].astype(str).str.strip().str.lower()
    unique_labels = sorted(raw.unique())
    print(f"[data] Unique raw label values: {unique_labels}")

    mapped = raw.map({k.lower(): v for k, v in label_map.items()})
    n_unknown = int(mapped.isna().sum())
    if n_unknown:
        print(
            f"[data] WARNING: dropping {n_unknown:,} rows with labels outside "
            f"{list(label_map.keys())}."
        )
    df = df.assign(**{label_col: mapped})
    df = df[df[label_col].notna()].copy()
    df[label_col] = df[label_col].astype(int)
    return df


def _clean_text(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    text_col = config["dataset"]["text_column"]
    df[text_col] = df[text_col].astype(str).str.strip()
    before = len(df)
    df = df[df[text_col].str.len() > 0].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[data] Dropped {dropped:,} rows with empty text.")
    return df


def split_dataframe(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:

    split_col = config["dataset"]["split_column"]
    split_map = config["dataset"]["split_map"]
    raw_values = df[split_col].astype(str).str.strip().str.lower()
    print(f"[data] Unique raw split values: {sorted(raw_values.unique())}")

    splits: Dict[str, pd.DataFrame] = {}
    claimed = pd.Series(False, index=df.index)
    for canonical, source_values in split_map.items():
        wanted = {str(v).strip().lower() for v in source_values}
        mask = raw_values.isin(wanted)
        splits[canonical] = df[mask].copy()
        claimed = claimed | mask

    unclaimed = int((~claimed).sum())
    if unclaimed:
        leftover = sorted(raw_values[~claimed].unique())
        print(
            f"[data] WARNING: {unclaimed:,} rows had unrecognized split values "
            f"{leftover} and were discarded."
        )

    for name, part in splits.items():
        if len(part) == 0:
            raise ValueError(
                f"Split '{name}' is empty after filtering. Check the "
                "`dataset.split_map` values in config.yaml against the unique "
                "raw split values printed above."
            )
    return splits


def summarize_splits(
    splits: Dict[str, pd.DataFrame], config: Dict[str, Any]
) -> Dict[str, Any]:
    label_col = config["dataset"]["label_column"]
    class_names = {v: k for k, v in config["dataset"]["label_map"].items()}

    summary: Dict[str, Any] = {"version": config["dataset"]["version"], "splits": {}}
    total = sum(len(p) for p in splits.values())
    for name, part in splits.items():
        counts = part[label_col].value_counts().to_dict()
        per_class = {
            class_names.get(int(k), str(k)): int(v) for k, v in counts.items()
        }
        n = len(part)
        hate_frac = (counts.get(1, 0) / n) if n else 0.0
        summary["splits"][name] = {
            "num_examples": n,
            "per_class": per_class,
            "hate_fraction": round(hate_frac, 4),
        }
    summary["total_examples"] = total
    return summary


def save_splits(
    splits: Dict[str, pd.DataFrame], config: Dict[str, Any]
) -> Dict[str, Path]:
    text_col = config["dataset"]["text_column"]
    label_col = config["dataset"]["label_column"]
    out_dir = resolve_path(config["paths"]["processed_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    for name, part in splits.items():
        out_path = out_dir / f"{name}.csv"
        part[[text_col, label_col]].to_csv(out_path, index=False, encoding="utf-8")
        paths[name] = out_path
        print(f"[data] Wrote {len(part):,} rows -> {out_path}")
    return paths

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf


def convert_saved_model(
    saved_model_dir: Path, allow_select_ops: bool = False
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if allow_select_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    return converter.convert()


def _convert_with_fallback(saved_model_dir: Path) -> Tuple[bytes, bool]:
    try:
        return convert_saved_model(saved_model_dir, allow_select_ops=False), False
    except Exception:  # noqa: BLE001 - retry path
        return convert_saved_model(saved_model_dir, allow_select_ops=True), True


def _make_converter(
    saved_model_dir: Path,
    mode: str,
    representative_dataset,
    allow_select_ops: bool,
):
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    if mode == "float32":
        pass
    elif mode == "dynamic_range":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        # Token-ID inputs stay integer; we do NOT force int8 I/O types.
    else:
        raise ValueError(f"Unknown export mode '{mode}'.")

    if allow_select_ops:
        ops = ops + [tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_ops = ops
    return converter


def convert_for_deployment(
    saved_model_dir: Path,
    mode: str = "float32",
    representative_dataset=None,
) -> tuple[bytes, bool]:
    try:
        model = _make_converter(
            saved_model_dir, mode, representative_dataset, allow_select_ops=False
        ).convert()
        return model, False
    except Exception:  # noqa: BLE001 - retry with Flex ops
        model = _make_converter(
            saved_model_dir, mode, representative_dataset, allow_select_ops=True
        ).convert()
        return model, True


def smoke_test_saved_model(
    saved_model_dir: Path,
    out_path: Path,
    label: str,
    seq_len: Optional[int] = None,
) -> bool:
    try:
        print(f"[smoke] {label}: converting to TFLite "
              "(can take a minute for large models)...", flush=True)
        tflite_model, used_select = _convert_with_fallback(saved_model_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(tflite_model)
        size_kb = len(tflite_model) / 1024.0
        interpreter = tf.lite.Interpreter(model_content=tflite_model)

        for det in interpreter.get_input_details():
            shape = list(det["shape"])
            resized: List[int] = []
            for i, dim in enumerate(shape):
                if i == 0:
                    resized.append(1)
                elif dim is None or int(dim) < 1:
                    resized.append(seq_len if seq_len else 1)
                else:
                    resized.append(int(dim))
            interpreter.resize_tensor_input(det["index"], resized, strict=False)
        interpreter.allocate_tensors()

        in_details = interpreter.get_input_details()
        for det in in_details:
            dummy = np.zeros(det["shape"], dtype=det["dtype"])
            interpreter.set_tensor(det["index"], dummy)
        interpreter.invoke()
        out_details = interpreter.get_output_details()
        out = interpreter.get_tensor(out_details[0]["index"])

        flex = " (required SELECT_TF_OPS)" if used_select else ""
        print(
            f"[smoke] {label}: OK{flex} | size={size_kb:.1f} KB | "
            f"{len(in_details)} input(s) | out_shape={tuple(out.shape)} -> {out_path}"
        )
        return True
    except Exception as exc:  # noqa: BLE001 - smoke test must never abort training
        print(f"[smoke] {label}: FAILED to convert/run -> {type(exc).__name__}: {exc}")
        return False

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_pair(
    history: Dict[str, List[float]],
    train_key: str,
    val_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> Path | None:
    if train_key not in history:
        return None
    epochs = range(1, len(history[train_key]) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history[train_key], label=f"train {ylabel}")
    if val_key in history:
        plt.plot(epochs, history[val_key], label=f"val {ylabel}")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def plot_training_history(history: Dict[str, List[float]], out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    loss_path = _plot_pair(
        history, "loss", "val_loss", "Training/validation loss", "loss",
        out_dir / "loss.png",
    )
    if loss_path:
        written.append(loss_path)

    acc_path = _plot_pair(
        history, "accuracy", "val_accuracy", "Training/validation accuracy",
        "accuracy", out_dir / "accuracy.png",
    )
    if acc_path:
        written.append(acc_path)

    for p in written:
        print(f"[metrics] Wrote plot -> {p}")
    return written


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    class_names: List[str],
) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "per_class": {name: report[name] for name in class_names},
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_names,
        "num_examples": int(len(y_true)),
    }

def metrics_at_threshold(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float
) -> Dict[str, float]:
    yt = np.asarray(y_true)
    yp = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    n = len(yt)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / n) if n else 0.0,
        "precision_hate": float(prec),
        "recall_hate": float(rec),
        "f1_hate": float(f1),
    }


def best_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    objective: str = "f1",
    steps: int = 201,
) -> Dict[str, float]:
    thresholds = np.linspace(0.0, 1.0, steps)
    best = {"threshold": 0.5, objective: -1.0}
    for t in thresholds:
        m = metrics_at_threshold(y_true, y_score, float(t))
        if m[objective] > best[objective]:
            best = {"threshold": float(t), objective: m[objective]}
    return best


def plot_precision_recall_curve(
    y_true: Sequence[int], y_score: Sequence[float], out_path: Path
) -> Path:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(5.5, 5))
    plt.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
    plt.xlabel("Recall (hate)")
    plt.ylabel("Precision (hate)")
    plt.title("Precision-Recall curve (positive class: hate)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[metrics] Wrote plot -> {out_path}")
    return out_path


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_names: List[str],
    out_path: Path,
    normalize: bool = False,
) -> Path:
    cm_arr = np.array(cm, dtype=float)
    title = "Confusion matrix"
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_arr = np.divide(cm_arr, row_sums, where=row_sums != 0)
        title = "Confusion matrix (row-normalized)"

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    thresh = cm_arr.max() / 2.0
    fmt = ".2f" if normalize else ".0f"
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[metrics] Wrote plot -> {out_path}")
    return out_path


def plot_roc_curve(
    y_true: Sequence[int], y_score: Sequence[float], out_path: Path
) -> Path:
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (positive class: hate)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[metrics] Wrote plot -> {out_path}")
    return out_path

import json
from typing import Any, Dict, List
import numpy as np
import tensorflow as tf
from tensorflow import keras


INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")


def _masked_mean_pool(args):
    sequence, mask = args
    mask = tf.cast(mask, sequence.dtype)[..., None]
    summed = tf.reduce_sum(sequence * mask, axis=1)
    counts = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-9)
    return summed / counts

_TOKENIZER_CACHE: Dict[str, Any] = {}

def _mb_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["model"]["mobilebert"]

# Tokenizer
def load_tokenizer(config: Dict[str, Any]):
    """Load (and cache) the WordPiece tokenizer for the configured model."""
    name = _mb_cfg(config)["hf_model_name"]
    if name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZER_CACHE[name]

def tokenize(config: Dict[str, Any], texts: List[str]) -> Dict[str, np.ndarray]:
    tokenizer = load_tokenizer(config)
    seq_len = _mb_cfg(config)["seq_len"]
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {name: enc[name].astype(np.int32) for name in INPUT_NAMES if name in enc}

def build_mobilebert_datasets(config: Dict[str, Any]) -> Dict[str, tf.data.Dataset]:
    """Build train/validation/test datasets of (input-dict, label) batches."""
    batch_size = config["training"]["batch_size"]
    seed = config["seed"]
    autotune = tf.data.AUTOTUNE

    datasets: Dict[str, tf.data.Dataset] = {}
    for split in ("train", "validation", "test"):
        texts, labels = load_split(config, split)
        print(f"[data] Tokenizing {split} ({len(texts):,} texts)...", flush=True)
        inputs = tokenize(config, texts)
        ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if split == "train":
            ds = ds.shuffle(
                buffer_size=len(labels), seed=seed, reshuffle_each_iteration=True
            )
        ds = ds.batch(batch_size).prefetch(autotune)
        datasets[split] = ds
    return datasets

# Encoder
def _load_tf_mobilebert(name: str):
    """Load TFMobileBertModel, falling back to PyTorch weights if needed."""
    from transformers import TFMobileBertModel

    try:
        return TFMobileBertModel.from_pretrained(name)
    except (OSError, EnvironmentError):
        try:
            return TFMobileBertModel.from_pretrained(name, from_pt=True)
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                f"Could not load TF weights for '{name}', and the PyTorch "
                "fallback failed. Install a CPU build of torch on the "
                "workstation (used offline only):\n"
                "  pip install torch --index-url "
                "https://download.pytorch.org/whl/cpu"
            ) from exc


def build_mobilebert_encoder(config: Dict[str, Any]) -> keras.Model:
    """Encoder: (input_ids, attention_mask, token_type_ids) -> pooled vector.

    Frozen on-device. ``fine_tune_encoder`` controls whether its weights are
    trainable during offline training.
    """
    mb = _mb_cfg(config)
    seq_len = mb["seq_len"]

    input_ids = keras.Input(shape=(seq_len,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(seq_len,), dtype="int32", name="attention_mask")
    token_type_ids = keras.Input(shape=(seq_len,), dtype="int32", name="token_type_ids")

    bert = _load_tf_mobilebert(mb["hf_model_name"])
    bert.trainable = bool(get("fine_tune_encoder", True))

    outputs = bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    sequence = outputs.last_hidden_state
    pooled = keras.layers.Lambda(_masked_mean_pool, name="masked_mean_pool")(
        [sequence, attention_mask]
    )
    normed = keras.layers.BatchNormalization(name="encoder_batchnorm")(pooled)
    return keras.Model(
        [input_ids, attention_mask, token_type_ids], normed, name="encoder"
    )


# Android artifacts (WordPiece vocab + tokenization examples)
def export_artifacts(config: Dict[str, Any], num_examples: int = 15) -> Dict[str, Any]:
    """Export vocab.txt / vocab.json, tokenization_examples.json, tokenizer files."""

    tokenizer = load_tokenizer(config)
    seq_len = _mb_cfg(config)["seq_len"]
    out_dir = resolve_path(config["paths"]["vectorizer_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    vocab_tokens = [tok for tok, _ in vocab_items]

    vocab_path = out_dir / VOCAB_FILENAME
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_tokens) + "\n")
    with open(out_dir / VOCAB_JSON_FILENAME, "w", encoding="utf-8") as f:
        json.dump(vocab_tokens, f, ensure_ascii=False)
    print(f"[prep] Wrote WordPiece vocabulary ({len(vocab_tokens):,}) -> {vocab_path}")

    val_texts, _ = load_split(config, "validation")
    sample_texts = list(val_texts[:num_examples]) + EDGE_CASE_TEXTS
    enc = tokenizer(
        sample_texts, padding="max_length", truncation=True, max_length=seq_len
    )
    do_lower = bool(getattr(tokenizer, "do_lower_case", True))
    examples = []
    for i, text in enumerate(sample_texts):
        ids = list(enc["input_ids"][i])
        examples.append(
            {
                "text": text,
                "tokens": tokenizer.convert_ids_to_tokens(ids),
                "input_ids": ids,
                "attention_mask": list(enc["attention_mask"][i]),
                "token_type_ids": list(enc["token_type_ids"][i]),
                "non_pad_count": int(sum(enc["attention_mask"][i])),
            }
        )

    payload = {
        "contract": {
            "tokenizer": "wordpiece",
            "hf_model_name": _mb_cfg(config)["hf_model_name"],
            "do_lower_case": do_lower,
            "max_len": seq_len,
            "pad_token_id": int(tokenizer.pad_token_id),
            "cls_token": tokenizer.cls_token,
            "sep_token": tokenizer.sep_token,
            "unk_token": tokenizer.unk_token,
            "note": (
                "WordPiece tokenization with [CLS] ... [SEP], right-padded with "
                "pad_token_id to max_len and truncated to max_len. vocab.txt line "
                "number (0-based) == token ID. The model takes input_ids, "
                "attention_mask and token_type_ids."
            ),
        },
        "examples": examples,
    }
    examples_path = out_dir / TOKENIZATION_EXAMPLES_FILENAME
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[prep] Wrote {len(examples)} tokenization examples -> {examples_path}")

    tok_dir = out_dir / "mobilebert_tokenizer"
    tokenizer.save_pretrained(str(tok_dir))
    print(f"[prep] Saved tokenizer files -> {tok_dir}")

    return {"vocab_size": len(vocab_tokens), "seq_len": seq_len}
# Model: a separable encoder + head for hate-speech classification.
# full = head(encoder(inputs))


from typing import Any, Dict, Tuple

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_encoder(config: Dict[str, Any]) -> keras.Model:
    pp = config["preprocessing"]
    m = config["model"]
    max_len = pp["max_len"]
    vocab_size = pp["vocab_size"]

    inputs = keras.Input(shape=(max_len,), dtype="int32", name="token_ids")
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=m["embedding_dim"],
        name="embedding",
    )(inputs)
    x = layers.Conv1D(
        filters=m["conv_filters"],
        kernel_size=m["conv_kernel_size"],
        padding="same",
        activation="relu",
        name="conv1d",
    )(x)
    outputs = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    return keras.Model(inputs, outputs, name="encoder")


# classification head
def build_head(
    config: Dict[str, Any], input_dim: int, logits: bool = False
) -> keras.Model:
    m = config["model"]
    inputs = keras.Input(shape=(input_dim,), dtype="float32", name="features")
    x = layers.Dropout(m["dropout"], name="head_dropout_1")(inputs)
    x = layers.Dense(m["head_hidden_units"], activation="relu", name="head_dense")(x)
    x = layers.Dropout(m["dropout"], name="head_dropout_2")(x)
    activation = None if logits else "softmax"
    outputs = layers.Dense(2, activation=activation, name="predictions")(x)
    return keras.Model(inputs, outputs, name="head")


def output_is_logits(config: Dict[str, Any]) -> bool:
    return config["model"].get("encoder_type", "cnn") == "mobilebert"


def build_encoder(config: Dict[str, Any]) -> keras.Model:
    encoder_type = config["model"].get("encoder_type", "cnn")
    if encoder_type == "cnn":
        return build_cnn_encoder(config)
    if encoder_type == "mobilebert":

        return build_mobilebert_encoder(config)
    raise ValueError(
        f"Unknown model.encoder_type '{encoder_type}'. Use 'cnn' or 'mobilebert'."
    )


def build_model(
    config: Dict[str, Any],
) -> Tuple[keras.Model, keras.Model, keras.Model]:
    encoder = build_encoder(config)
    encoder_output_dim = encoder.output_shape[-1]
    head = build_head(config, encoder_output_dim, logits=output_is_logits(config))

    fresh_inputs = [
        keras.Input(shape=inp.shape[1:], dtype=inp.dtype, name=name)
        for inp, name in zip(encoder.inputs, encoder.input_names)
    ]
    encoder_call_arg = fresh_inputs if len(fresh_inputs) > 1 else fresh_inputs[0]
    outputs = head(encoder(encoder_call_arg))
    full_model = keras.Model(fresh_inputs, outputs, name="hate_classifier")
    return full_model, encoder, head
from typing import Any, Dict, List

EDGE_CASE_TEXTS = [
    "HELLO World",
    "this contains an obviously unknown token zzqxwv",
    "don't, stop! really?",
    "multiple    spaces\tand\ttabs here",
]


def _encoder_type(config: Dict[str, Any]) -> str:
    return config["model"].get("encoder_type", "cnn")


def build_datasets(config: Dict[str, Any]) -> Dict[str, Any]:
    et = _encoder_type(config)
    if et == "cnn":

        vectorizer = load_vectorizer(config)
        return build_tf_datasets(config, vectorizer)
    if et == "mobilebert":

        return build_mobilebert_datasets(config)
    raise ValueError(f"Unknown model.encoder_type '{et}'.")


def texts_to_inputs(config: Dict[str, Any], texts: List[str]):
    et = _encoder_type(config)
    if et == "cnn":
        import numpy as np
        import tensorflow as tf


        vectorizer = load_vectorizer(config)
        return vectorizer(tf.constant(texts)).numpy().astype(np.int32)
    if et == "mobilebert":

        return tokenize(config, texts)
    raise ValueError(f"Unknown model.encoder_type '{et}'.")


def export_encoder_artifacts(
    config: Dict[str, Any], num_examples: int = 15
) -> Dict[str, Any]:
    et = _encoder_type(config)
    if et == "cnn":
        return _export_cnn_artifacts(config, num_examples)
    if et == "mobilebert":

        return export_artifacts(config, num_examples)
    raise ValueError(f"Unknown model.encoder_type '{et}'.")


def _export_cnn_artifacts(config: Dict[str, Any], num_examples: int) -> Dict[str, Any]:

    train_texts, _ = load_split(config, "train")
    print(f"[prep] Adapting vectorizer on {len(train_texts):,} train texts...")
    vectorizer = build_vectorizer(config)
    adapt_vectorizer(vectorizer, train_texts)
    save_vectorizer(vectorizer, config)

    val_texts, _ = load_split(config, "validation")
    oov_rate = compute_oov_rate(vectorizer, val_texts)
    actual_vocab_size = len(vectorizer.get_vocabulary(include_special_tokens=True))
    print(f"[prep] Adapted vocabulary size: {actual_vocab_size:,}")
    print(f"[prep] OOV rate on validation:  {oov_rate:.4%}")

    sample_texts = val_texts[:num_examples] + EDGE_CASE_TEXTS
    export_tokenization_examples(vectorizer, config, sample_texts)
    return {"vocab_size": actual_vocab_size, "oov_rate_validation": oov_rate}

# Phase B - on-device training: a trainable head with LiteRT signatures.
#  - The TRAINABLE HEAD is wrapped here in a tf.Module exposing four LiteRT
#    signatures - train / infer / save / restore - that operate on those feature
#    vectors. On device: collect (embedding, user_label) from feedback, call
#    `train`; predict via `infer`; persist personalized weights via `save`/
#    `restore`.


from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf


class HeadOnDeviceTrainer(tf.Module):

    def __init__(self, head: tf.keras.Model, learning_rate: float):
        super().__init__()
        self.head = head
        self.lr = tf.constant(learning_rate, dtype=tf.float32)

    @tf.function
    def train(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.head(x, training=False)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y, logits, from_logits=True
                )
            )
        trainable = self.head.trainable_variables
        grads = tape.gradient(loss, trainable)
        for var, grad in zip(trainable, grads):
            var.assign_sub(self.lr * grad)
        return {"loss": loss}

    @tf.function
    def infer(self, x):
        logits = self.head(x, training=False)
        return {"probabilities": tf.nn.softmax(logits, axis=-1)}

    @tf.function
    def save(self, unused):
        del unused
        flat = tf.concat(
            [tf.reshape(v, [-1]) for v in self.head.trainable_variables], axis=0
        )
        return {"weights": flat}

    @tf.function
    def restore(self, weights):
        offset = 0
        for var in self.head.trainable_variables:
            n = int(np.prod(var.shape.as_list()))
            var.assign(tf.reshape(weights[offset : offset + n], var.shape))
            offset += n
        return {"restored": tf.constant(True)}


def build_head_trainer_tflite(
    head: tf.keras.Model, emb_dim: int, learning_rate: float, saved_dir: Path
) -> bytes:
    module = HeadOnDeviceTrainer(head, learning_rate)
    total_params = int(
        sum(np.prod(v.shape.as_list()) for v in head.trainable_variables)
    )

    signatures = {
        "train": module.train.get_concrete_function(
            tf.TensorSpec([None, emb_dim], tf.float32, name="x"),
            tf.TensorSpec([None], tf.int32, name="y"),
        ),
        "infer": module.infer.get_concrete_function(
            tf.TensorSpec([None, emb_dim], tf.float32, name="x"),
        ),
        "save": module.save.get_concrete_function(
            tf.TensorSpec([1], tf.float32, name="unused"),
        ),
        "restore": module.restore.get_concrete_function(
            tf.TensorSpec([total_params], tf.float32, name="weights"),
        ),
    }

    saved_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(module, str(saved_dir), signatures=signatures)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,  # required for the training ops
    ]
    converter.experimental_enable_resource_variables = True
    return converter.convert()


def verify_head_trainer(
    tflite_bytes: bytes, emb_dim: int, train_steps: int = 60
) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    results: Dict[str, Any] = {
        "signatures": sorted(interpreter.get_signature_list().keys())
    }

    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, emb_dim)).astype(np.float32)
    y = rng.integers(0, 2, size=8).astype(np.int32)

    infer = interpreter.get_signature_runner("infer")
    probs = infer(x=x)["probabilities"]
    results["infer_output_shape"] = list(probs.shape)

    train = interpreter.get_signature_runner("train")
    loss_first = float(train(x=x, y=y)["loss"])
    loss_last = loss_first
    for _ in range(train_steps - 1):
        loss_last = float(train(x=x, y=y)["loss"])
    results["train_loss_first"] = round(loss_first, 4)
    results["train_loss_last"] = round(loss_last, 4)
    results["train_loss_decreased"] = bool(loss_last < loss_first)

    try:
        save = interpreter.get_signature_runner("save")
        restore = interpreter.get_signature_runner("restore")
        weights = save(unused=np.zeros([1], dtype=np.float32))["weights"]
        results["num_head_params"] = int(weights.shape[0])
        probs_saved = infer(x=x)["probabilities"]
        for _ in range(10):
            train(x=x, y=y)
        restore(weights=weights)
        probs_restored = infer(x=x)["probabilities"]
        results["save_restore"] = "ok"
        results["restore_recovers_state"] = bool(
            np.allclose(probs_saved, probs_restored, atol=1e-4)
        )
    except Exception as exc:
        results["save_restore"] = f"failed ({type(exc).__name__}: {exc})"

    return results
