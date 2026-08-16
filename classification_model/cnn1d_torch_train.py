from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataclasses import fields as _dataclass_fields

from classification_model.model_zoo.cnn1d_torch_model import (
    Conv1DCNN,
    Conv1DHyperparams,
    NumpyLabelEncoder,
    SegmentDataset,
    compute_train_zscore_config,
    infer_device,
    make_normalization_config,
    normalize_segments,
    set_random_seed,
)


REQUIRED_GT_COLUMNS = {"source", "onset_s", "offset_s", "label"}


@dataclass
class PhraseSamplingConfig:
    sample_rate: Optional[int] = None
    sample_secs: float = 0.05
    samples_per_phrase: int = 8
    pad_mode: str = "constant"
    normalize_waveform: bool = True
    include_labels: Optional[Sequence[str]] = None
    exclude_labels: Sequence[str] = ()
    min_phrase_duration_s: float = 0.0
    phrase_gap_tolerance_s: float = 0.0
    max_phrases_per_label: Optional[int] = None
    rng_seed: int = 13
    verbose: bool = True


class PhraseSegmentDatasetBuilder:
    def __init__(self, wav_dir: Path, gt_df: pd.DataFrame, cfg: PhraseSamplingConfig):
        self.wav_dir = Path(wav_dir)
        self.gt_df = self._validate_gt(gt_df)
        self.cfg = cfg
        if not self.wav_dir.exists():
            raise FileNotFoundError(f"wav_dir does not exist: {self.wav_dir}")
        self.rng = np.random.default_rng(cfg.rng_seed)

    @staticmethod
    def _validate_gt(gt_df: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_GT_COLUMNS.difference(gt_df.columns)
        if missing:
            raise ValueError(f"Ground-truth dataframe is missing required columns: {sorted(missing)}")
        out = gt_df.copy()
        out["source"] = out["source"].astype(str)
        out["label"] = out["label"].astype(str)
        out["onset_s"] = out["onset_s"].astype(float)
        out["offset_s"] = out["offset_s"].astype(float)
        out = out[out["offset_s"] > out["onset_s"]].copy()
        out = out.sort_values(["source", "onset_s", "offset_s"]).reset_index(drop=True)
        return out

    def _print(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg)

    def _filter_gt(self) -> pd.DataFrame:
        gt = self.gt_df.copy()
        if self.cfg.include_labels is not None:
            allowed = {str(x) for x in self.cfg.include_labels}
            gt = gt[gt["label"].isin(allowed)].copy()
        if self.cfg.exclude_labels:
            excluded = {str(x) for x in self.cfg.exclude_labels}
            gt = gt[~gt["label"].isin(excluded)].copy()
        gt = gt.reset_index(drop=True)
        self._print(f"[builder] Ground-truth rows after filtering: {len(gt)}")
        self._print(f"[builder] Labels after filtering: {sorted(gt['label'].unique().tolist())}")
        return gt

    def _resolve_wav_path(self, source: str) -> Path:
        wav_path = self.wav_dir / str(source)
        if wav_path.exists():
            return wav_path
        alt = Path(str(source))
        if alt.exists():
            return alt
        #raise FileNotFoundError(f"wav file was not found for source: {source}")

    @staticmethod
    def _normalize_segment(seg: np.ndarray) -> np.ndarray:
        seg = seg.astype(np.float32)
        seg = seg - np.mean(seg)
        peak = np.max(np.abs(seg))
        if peak > 0:
            seg = seg / peak
        return seg

    def _extract_centered_window(self, wav: np.ndarray, sr: int, center_s: float, num_samples: int) -> np.ndarray:
        center_sample = int(round(center_s * sr))
        start_sample = center_sample - (num_samples // 2)
        end_sample = start_sample + num_samples
        left_pad = max(0, -start_sample)
        right_pad = max(0, end_sample - len(wav))
        clipped_start = max(0, start_sample)
        clipped_end = min(len(wav), end_sample)
        seg = wav[clipped_start:clipped_end]
        if left_pad or right_pad:
            seg = np.pad(seg, (left_pad, right_pad), mode=self.cfg.pad_mode)
        if len(seg) != num_samples:
            if len(seg) > num_samples:
                seg = seg[:num_samples]
            else:
                seg = np.pad(seg, (0, num_samples - len(seg)), mode=self.cfg.pad_mode)
        if self.cfg.normalize_waveform:
            seg = self._normalize_segment(seg)
        return seg.astype(np.float32)

    def build_phrase_table(self) -> pd.DataFrame:
        gt = self._filter_gt()
        phrase_rows: List[Dict] = []
        phrase_global_id = 0

        for source, grp in gt.groupby("source", sort=False):
            grp = grp.sort_values(["onset_s", "offset_s"]).reset_index(drop=True)
            current = None
            phrase_idx_in_source = 0
            prev_offset = None
            prev_label = None

            for rec in grp.itertuples(index=False):
                same_label = prev_label == rec.label
                close_enough = prev_offset is not None and (float(rec.onset_s) - float(prev_offset)) <= self.cfg.phrase_gap_tolerance_s
                should_merge = current is not None and same_label and close_enough

                if should_merge:
                    current["offset_s"] = max(float(current["offset_s"]), float(rec.offset_s))
                    current["n_syllables"] += 1
                else:
                    if current is not None:
                        current["duration_s"] = float(current["offset_s"] - current["onset_s"])
                        if current["duration_s"] >= self.cfg.min_phrase_duration_s:
                            phrase_rows.append(current)
                    current = {
                        "source": str(source),
                        "phrase_id": int(phrase_global_id),
                        "phrase_index_in_source": int(phrase_idx_in_source),
                        "label": str(rec.label),
                        "onset_s": float(rec.onset_s),
                        "offset_s": float(rec.offset_s),
                        "n_syllables": 1,
                    }
                    phrase_global_id += 1
                    phrase_idx_in_source += 1

                prev_offset = float(rec.offset_s)
                prev_label = str(rec.label)

            if current is not None:
                current["duration_s"] = float(current["offset_s"] - current["onset_s"])
                if current["duration_s"] >= self.cfg.min_phrase_duration_s:
                    phrase_rows.append(current)

        phrase_df = pd.DataFrame(phrase_rows)
        if phrase_df.empty:
            raise ValueError("No phrases remain after merging and filtering")

        if self.cfg.max_phrases_per_label is not None:
            kept = []
            for label, grp in phrase_df.groupby("label", sort=False):
                if len(grp) > self.cfg.max_phrases_per_label:
                    take_idx = self.rng.choice(grp.index.to_numpy(), size=self.cfg.max_phrases_per_label, replace=False)
                    kept.append(phrase_df.loc[np.sort(take_idx)])
                else:
                    kept.append(grp)
            phrase_df = pd.concat(kept, axis=0).sort_values(["source", "onset_s", "offset_s"]).reset_index(drop=True)

        self._print(f"[builder] Built {len(phrase_df)} phrases from {phrase_df['source'].nunique()} wav files")
        counts = phrase_df["label"].value_counts().sort_index().to_dict()
        self._print(f"[builder] Phrase counts by label: {counts}")
        return phrase_df

    def build_dataset_payload(self) -> Dict:
        phrase_df = self.build_phrase_table()
        wav_cache: Dict[str, Tuple[np.ndarray, int, int]] = {}
        segments: List[np.ndarray] = []
        labels: List[str] = []
        sources: List[str] = []
        phrase_ids: List[int] = []
        phrase_indices: List[int] = []
        center_s_list: List[float] = []
        phrase_onsets: List[float] = []
        phrase_offsets: List[float] = []
        phrase_durations: List[float] = []
        phrase_n_syllables: List[int] = []

        self._print("[builder] Extracting random phrase-centered samples")
        for i, rec in enumerate(phrase_df.itertuples(index=False), start=1):
            source = str(rec.source)
            if source not in wav_cache:
                wav_path = self._resolve_wav_path(source)
                try:
                    wav, sr = sf.read(wav_path)
                except Exception as exc:
                    warnings.warn(f"Skipping {source} because wav could not be read: {exc}")
                    continue
                if wav.ndim > 1:
                    wav = wav[:, np.argmax(np.std(wav, axis=0))]
                wav = wav.astype(np.float32)
                if self.cfg.sample_rate is not None and int(sr) != int(self.cfg.sample_rate):
                    raise ValueError(
                        f"Sample-rate mismatch for {wav_path.name}: found {sr}, expected {self.cfg.sample_rate}. "
                        "Please resample upstream or set sample_rate=None."
                    )
                wav_cache[source] = (wav, int(sr), len(wav))
                self._print(f"[builder] Loaded {wav_path.name} | sr={sr} | n={len(wav)}")

            wav, sr, _ = wav_cache[source]
            num_samples = max(1, int(round(self.cfg.sample_secs * sr)))
            low = float(rec.onset_s)
            high = float(rec.offset_s)
            if high <= low:
                continue
            centers = self.rng.uniform(low=low, high=high, size=self.cfg.samples_per_phrase)
            for center_s in centers:
                seg = self._extract_centered_window(wav=wav, sr=sr, center_s=float(center_s), num_samples=num_samples)
                segments.append(seg)
                labels.append(str(rec.label))
                sources.append(source)
                phrase_ids.append(int(rec.phrase_id))
                phrase_indices.append(int(rec.phrase_index_in_source))
                center_s_list.append(float(center_s))
                phrase_onsets.append(float(rec.onset_s))
                phrase_offsets.append(float(rec.offset_s))
                phrase_durations.append(float(rec.duration_s))
                phrase_n_syllables.append(int(rec.n_syllables))

            if self.cfg.verbose and (i % 100 == 0 or i == len(phrase_df)):
                self._print(f"[builder] Progress: phrases {i}/{len(phrase_df)} | samples so far={len(segments)}")

        if not segments:
            raise ValueError("No segments were generated")

        sample_rates = {sr for _, sr, _ in wav_cache.values()}
        if len(sample_rates) != 1:
            raise ValueError(
                f"Dataset payload currently expects a single sample rate across files, got: {sorted(sample_rates)}"
            )
        sample_rate = int(next(iter(sample_rates)))
        num_samples = int(round(self.cfg.sample_secs * sample_rate))

        X = np.stack(segments).astype(np.float32)
        payload = {
            "X": X,
            "labels": np.asarray(labels, dtype=object),
            "sources": np.asarray(sources, dtype=object),
            "phrase_ids": np.asarray(phrase_ids, dtype=np.int64),
            "phrase_indices": np.asarray(phrase_indices, dtype=np.int64),
            "center_s": np.asarray(center_s_list, dtype=np.float32),
            "phrase_onset_s": np.asarray(phrase_onsets, dtype=np.float32),
            "phrase_offset_s": np.asarray(phrase_offsets, dtype=np.float32),
            "phrase_duration_s": np.asarray(phrase_durations, dtype=np.float32),
            "phrase_n_syllables": np.asarray(phrase_n_syllables, dtype=np.int64),
            "sample_rate": sample_rate,
            "sample_secs": float(self.cfg.sample_secs),
            "num_samples": num_samples,
            "config": asdict(self.cfg),
            "phrase_table": phrase_df,
            "summary": {
                "n_samples": int(len(X)),
                "n_phrases": int(len(phrase_df)),
                "n_sources": int(phrase_df["source"].nunique()),
                "label_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
                "samples_per_phrase": int(self.cfg.samples_per_phrase),
            },
        }
        self._print(f"[builder] Final dataset: X.shape={X.shape}")
        return payload

    def save_dataset_pickle(self, out_path: Path) -> Dict:
        payload = self.build_dataset_payload()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        sidecar = out_path.with_suffix(out_path.suffix + ".summary.json")
        sidecar.write_text(json.dumps(payload["summary"], indent=2), encoding="utf-8")
        self._print(f"[builder] Saved dataset pickle to: {out_path}")
        return payload


class Conv1DClassifier:
    def __init__(self, hp: Optional[Conv1DHyperparams] = None, device: Optional[torch.device] = None, verbose: bool = True):
        self.hp = hp or Conv1DHyperparams()
        self.device = device or infer_device()
        self.verbose = verbose
        self.label_encoder = NumpyLabelEncoder()
        self.model: Optional[Conv1DCNN] = None
        self.history: List[Dict] = []
        self.normalization_config: Dict = make_normalization_config(mode="none")

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _build_model(self, input_num_samples: int, n_classes: int) -> Conv1DCNN:
        return Conv1DCNN(input_num_samples=input_num_samples, n_classes=n_classes, hp=self.hp).to(self.device)

    def _make_fit_normalization_config(self, X_train: np.ndarray) -> Dict:
        if not self.hp.normalize_waveform:
            return make_normalization_config(mode="none", eps=self.hp.normalization_eps)
        if self.hp.normalization_mode == "train_zscore":
            return compute_train_zscore_config(X_train, eps=self.hp.normalization_eps)
        return make_normalization_config(mode=self.hp.normalization_mode, eps=self.hp.normalization_eps)

    def _preprocess_segments(self, segments: np.ndarray) -> np.ndarray:
        return normalize_segments(np.asarray(segments, dtype=np.float32), self.normalization_config)

    def _compute_class_weights(self, y_train: np.ndarray) -> Optional[torch.Tensor]:
        if self.hp.class_weighting != "balanced":
            return None
        counts = np.bincount(y_train)
        weights = np.zeros_like(counts, dtype=np.float32)
        nonzero = counts > 0
        weights[nonzero] = len(y_train) / (len(counts) * counts[nonzero])
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def fit(self, train_payload: Dict, val_payload: Dict) -> List[Dict]:
        set_random_seed(self.hp.random_seed)
        X_train = np.asarray(train_payload["X"], dtype=np.float32)
        y_train_labels = np.asarray(train_payload["labels"], dtype=str)
        X_val = np.asarray(val_payload["X"], dtype=np.float32)
        y_val_labels = np.asarray(val_payload["labels"], dtype=str)

        self.normalization_config = self._make_fit_normalization_config(X_train)
        X_train = self._preprocess_segments(X_train)
        X_val = self._preprocess_segments(X_val)
        self._print(f"[normalization] {self.normalization_config}")

        self.label_encoder.fit(y_train_labels)
        y_train = self.label_encoder.transform(y_train_labels)
        y_val = self.label_encoder.transform(y_val_labels)

        self.model = self._build_model(input_num_samples=X_train.shape[1], n_classes=len(self.label_encoder.classes_))
        criterion = torch.nn.CrossEntropyLoss(weight=self._compute_class_weights(y_train))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp.learning_rate, weight_decay=self.hp.weight_decay)

        train_loader = DataLoader(SegmentDataset(X_train, y_train), batch_size=self.hp.batch_size, shuffle=True)
        val_loader = DataLoader(SegmentDataset(X_val, y_val), batch_size=self.hp.batch_size, shuffle=False)

        self._print(f"[train] device={self.device} | train={len(X_train)} | val={len(X_val)} | classes={len(self.label_encoder.classes_)}")
        self._print(f"[train] input_num_samples={X_train.shape[1]} | train_batches={len(train_loader)} | val_batches={len(val_loader)}")

        best_state = None
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        self.history = []

        for epoch in range(1, self.hp.epochs + 1):
            self.model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * len(yb)
                preds = logits.argmax(dim=1)
                running_correct += int((preds == yb).sum().item())
                running_total += len(yb)

                if self.verbose and (batch_idx % 20 == 0 or batch_idx == len(train_loader)):
                    self._print(
                        f"[train] epoch {epoch}/{self.hp.epochs} | batch {batch_idx}/{len(train_loader)} | "
                        f"loss={running_loss / max(running_total, 1):.4f} | acc={running_correct / max(running_total, 1):.4f}"
                    )

            train_loss = running_loss / max(running_total, 1)
            train_acc = running_correct / max(running_total, 1)

            self.model.eval()
            val_loss_total = 0.0
            val_total = 0
            val_preds = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_loss_total += float(loss.item()) * len(yb)
                    val_total += len(yb)
                    val_preds.append(logits.argmax(dim=1).cpu().numpy())

            y_val_pred = np.concatenate(val_preds) if val_preds else np.empty((0,), dtype=np.int64)
            val_loss = val_loss_total / max(val_total, 1)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
            }
            self.history.append(epoch_record)
            self._print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
                self._print(f"[train] New best model at epoch {epoch}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.hp.early_stopping_patience:
                    self._print(f"[train] Early stopping triggered at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self._print(f"[train] Restored best model from epoch {best_epoch}")
        return self.history

    def predict_proba(self, segments: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        batch_size = batch_size or self.hp.batch_size
        X = self._preprocess_segments(np.asarray(segments, dtype=np.float32))
        ds = SegmentDataset(X, np.zeros(len(X), dtype=np.int64))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        # Force eval mode so BatchNorm uses running stats and Dropout is disabled.
        # Previously this relied on the caller (main._prewarm_models) having set eval().
        self.model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, segments: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(segments)
        idx = probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(idx)

    def evaluate_payload(self, payload: Dict) -> Dict:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        X = self._preprocess_segments(np.asarray(payload["X"], dtype=np.float32))
        y_true = self.label_encoder.transform(np.asarray(payload["labels"], dtype=str))
        ds = SegmentDataset(X, y_true)
        loader = DataLoader(ds, batch_size=self.hp.batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.eval()
        total_loss = 0.0
        total_n = 0
        preds = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                total_loss += float(loss.item()) * len(yb)
                total_n += len(yb)
                preds.append(logits.argmax(dim=1).cpu().numpy())

        y_pred = np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.label_encoder.classes_.tolist(),
            zero_division=0,
            output_dict=True,
        )
        return {
            "loss": total_loss / max(total_n, 1),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "y_true": y_true,
            "y_pred": y_pred,
            "labels": self.label_encoder.classes_.tolist(),
        }

    def save(self, out_dir: Path, metadata: Optional[Dict] = None, input_num_samples: Optional[int] = None) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "cnn1d_model.pt")
        self.label_encoder.save(out_dir / "label_encoder.json")
        (out_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        meta = dict(metadata or {})
        if input_num_samples is not None:
            meta.setdefault("input_num_samples", int(input_num_samples))
        meta.setdefault("normalization_config", self.normalization_config)
        payload = {"hyperparams": asdict(self.hp), "metadata": meta}
        (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path, input_num_samples: Optional[int] = None, device: Optional[torch.device] = None) -> "Conv1DClassifier":
        payload = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
        # Tolerate hyperparam fields this (older) runtime does not define, e.g.
        # normalization_mode / normalization_eps written by newer tweetynet_flow.
        known_fields = {f.name for f in _dataclass_fields(Conv1DHyperparams)}
        hp_payload = {k: v for k, v in dict(payload["hyperparams"]).items() if k in known_fields}
        hp = Conv1DHyperparams(**hp_payload)
        obj = cls(hp=hp, device=device)
        obj.label_encoder = NumpyLabelEncoder.load(Path(model_dir) / "label_encoder.json")
        saved_meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        # Normalization is the classifier's responsibility at inference time. Recover the
        # exact config the model was trained with; fall back to the legacy convention
        # (segment_peak when normalize_waveform, else none) for models saved before
        # normalization_config existed.
        obj.normalization_config = saved_meta.get(
            "normalization_config",
            make_normalization_config(mode="segment_peak" if hp.normalize_waveform else "none", eps=hp.normalization_eps),
        )
        if input_num_samples is None:
            input_num_samples = saved_meta.get("input_num_samples")
        if input_num_samples is None:
            raise ValueError(
                "input_num_samples was not provided and could not be recovered from metadata.json. "
                "Please pass the exact number of samples used during training, or resave the model with updated metadata."
            )
        obj.model = Conv1DCNN(int(input_num_samples), len(obj.label_encoder.classes_), hp).to(obj.device)
        state = torch.load(Path(model_dir) / "cnn1d_model.pt", map_location=obj.device)
        obj.model.load_state_dict(state)
        return obj


def load_dataset_pickle(pickle_path: Path) -> Dict:
    with Path(pickle_path).open("rb") as f:
        payload = pickle.load(f)
    required = {"X", "labels", "sources", "sample_rate", "sample_secs", "num_samples"}
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Dataset pickle is missing required keys: {sorted(missing)}")
    return payload


def _slice_payload(payload: Dict, mask: np.ndarray) -> Dict:
    out = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            out[key] = value[mask]
        else:
            out[key] = value
    return out


def split_dataset_payload(
    payload: Dict,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 13,
    explicit_test_sources: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Tuple[Dict, Dict, Dict]:
    sources = np.asarray(payload["sources"], dtype=str)
    unique_sources = np.unique(sources)
    if len(unique_sources) < 3 and not explicit_test_sources:
        raise ValueError("Need at least 3 wav sources for train/val/test splitting")

    explicit_test = {str(x) for x in explicit_test_sources} if explicit_test_sources else set()
    test_sources = np.asarray(sorted(explicit_test.intersection(set(unique_sources))), dtype=object)
    remaining_sources = np.asarray([s for s in unique_sources if s not in explicit_test], dtype=object)

    if len(test_sources) == 0:
        train_val_sources, test_sources = train_test_split(remaining_sources, test_size=test_size, random_state=random_seed)
    else:
        train_val_sources = remaining_sources

    if len(train_val_sources) < 2:
        raise ValueError("Need at least 2 non-test wav sources for train/validation splitting")

    train_sources, val_sources = train_test_split(train_val_sources, test_size=val_size, random_state=random_seed)

    train_mask = np.isin(sources, train_sources)
    val_mask = np.isin(sources, val_sources)
    test_mask = np.isin(sources, test_sources)

    if verbose:
        print(f"[split] Train wavs: {len(train_sources)} | Val wavs: {len(val_sources)} | Test wavs: {len(test_sources)}")
        print(f"[split] Train samples: {int(train_mask.sum())} | Val samples: {int(val_mask.sum())} | Test samples: {int(test_mask.sum())}")

    return _slice_payload(payload, train_mask), _slice_payload(payload, val_mask), _slice_payload(payload, test_mask)


def build_phrase_dataset_pickle(
    wav_dir: Path,
    gt_df: pd.DataFrame,
    out_pickle_path: Path,
    cfg: Optional[PhraseSamplingConfig] = None,
) -> Dict:
    cfg = cfg or PhraseSamplingConfig()
    builder = PhraseSegmentDatasetBuilder(wav_dir=wav_dir, gt_df=gt_df, cfg=cfg)
    return builder.save_dataset_pickle(out_pickle_path)


def fit_from_dataset_pickle(
    dataset_pickle_path: Path,
    out_dir: Path,
    hp: Optional[Conv1DHyperparams] = None,
    explicit_test_sources: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict:
    hp = hp or Conv1DHyperparams()
    payload = load_dataset_pickle(dataset_pickle_path)
    if hp.sample_secs is None:
        hp.sample_secs = float(payload["sample_secs"])

    train_payload, val_payload, test_payload = split_dataset_payload(
        payload,
        random_seed=hp.random_seed,
        explicit_test_sources=explicit_test_sources,
        verbose=verbose,
    )

    clf = Conv1DClassifier(hp=hp, verbose=verbose)
    clf.fit(train_payload, val_payload)
    val_metrics = clf.evaluate_payload(val_payload)
    test_metrics = clf.evaluate_payload(test_payload) if len(test_payload["X"]) else None

    metadata = {
        "dataset_pickle_path": str(Path(dataset_pickle_path).resolve()),
        "n_train": int(len(train_payload["X"])),
        "n_val": int(len(val_payload["X"])),
        "n_test": int(len(test_payload["X"])),
        "labels": clf.label_encoder.classes_.tolist(),
        "sample_secs": float(payload["sample_secs"]),
        "num_samples": int(payload["num_samples"]),
        "sample_rate": int(payload["sample_rate"]),
        "n_sources": int(len(np.unique(payload["sources"]))),
    }
    clf.save(out_dir, metadata=metadata, input_num_samples=int(payload["num_samples"]))

    summary = {
        "metadata": metadata,
        "val_metrics": {k: v for k, v in val_metrics.items() if k not in {"y_true", "y_pred", "confusion_matrix", "classification_report"}},
        "test_metrics": None if test_metrics is None else {k: v for k, v in test_metrics.items() if k not in {"y_true", "y_pred", "confusion_matrix", "classification_report"}},
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "classifier": clf,
        "dataset_payload": payload,
        "train_payload": train_payload,
        "val_payload": val_payload,
        "test_payload": test_payload,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "metadata": metadata,
    }
