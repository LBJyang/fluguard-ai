"""
FluGuard Audio Engine
流感卫士 音频引擎

1. CoughDetector  — YAMNet (TF Hub) + fine-tuned keras head → cough probability
2. VoiceprintEngine — resemblyzer GE2E speaker encoder → enroll / verify
"""

import io
import logging
import numpy as np
from pathlib import Path

log = logging.getLogger("fluguard.audio")

# Path to the trained classifier head (relative to this file)
CLASSIFIER_PATH = Path(__file__).parent / "models" / "best_cough_classifier.keras"
SAMPLE_RATE = 16000
COUGH_THRESHOLD = 0.50   # standard threshold; v2 model has Recall 99.33% / Precision 96.75% on test set
VOICEPRINT_THRESHOLD = 0.75   # cosine similarity to accept a match (raised from 0.72)
PROFILES_PATH = Path(__file__).parent / "voiceprint_profiles.npz"  # disk persistence


# ─── Cough Detector ──────────────────────────────────────────────────────────

class CoughDetector:
    """
    Pipeline:
        raw audio bytes
            → librosa (16 kHz mono, normalised)
            → YAMNet (frozen, TF Hub)  →  1024-dim frame embeddings, averaged
            → trained Dense head       →  cough probability in [0, 1]
    """

    def __init__(self, model_path: Path = CLASSIFIER_PATH):
        self._yamnet = None
        self._classifier = None
        self._load(model_path)

    def _load(self, model_path: Path):
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            log.info("Loading YAMNet from TF Hub (cached after first download)...")
            self._yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            log.info("YAMNet loaded")

            if not model_path.exists():
                log.error(f"Classifier not found: {model_path}")
                return
            log.info(f"Loading cough classifier: {model_path.name}")
            self._classifier = tf.keras.models.load_model(str(model_path))
            log.info("CoughDetector ready  ✓")
        except Exception as exc:
            log.error(f"CoughDetector init failed: {exc}")

    @property
    def ready(self) -> bool:
        return self._yamnet is not None and self._classifier is not None

    def detect(self, audio_bytes: bytes) -> dict:
        """
        Detect cough in raw audio bytes (wav / webm / mp3 / ogg all accepted).

        Returns:
            {
              "probability": float,   # 0.0–1.0
              "is_cough":   bool,
              "label":      "cough" | "not_cough",
              "confidence": str,      # e.g. "94.2%"
            }
        """
        if not self.ready:
            return {"error": "CoughDetector not initialised"}

        try:
            import tensorflow as tf
            import librosa

            wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
            wav = wav.astype(np.float32)

            if np.max(np.abs(wav)) < 1e-6:
                return {"error": "Audio is silent"}

            # 去除首尾静音，避免大量静音帧稀释咳嗽 embedding
            # Trim leading/trailing silence so cough frames aren't diluted
            # top_db=30: slightly lenient — only remove near-silence, not quiet sounds
            wav_trimmed, _ = librosa.effects.trim(wav, top_db=30)
            # 如果裁剪后太短（<0.3s），保留原始音频
            wav = wav_trimmed if len(wav_trimmed) > SAMPLE_RATE * 0.3 else wav

            # ⚠️  Do NOT normalise here — the classifier was trained on librosa.load output
            # without forced normalisation. Forcing max-amplitude on every input shifts
            # the YAMNet embedding distribution and causes false positives on all non-silent audio.

            # YAMNet → (scores, embeddings, spectrogram)
            scores, embeddings, _ = self._yamnet(wav)

            # 与训练时保持一致：对时间帧取均值 → 固定 1024 维向量
            # Consistent with training: average over time frames → fixed 1024-dim vector
            embedding = tf.reduce_mean(embeddings, axis=0).numpy()   # (1024,)

            # 调试日志：YAMNet 自己的 top-1 类是什么
            mean_scores = tf.reduce_mean(scores, axis=0).numpy()
            top_class   = int(np.argmax(mean_scores))
            top_score   = float(mean_scores[top_class])
            log.info(f"YAMNet top_class={top_class} top_score={top_score:.3f} cough_score={float(mean_scores[42]):.3f}")

            prob = float(
                self._classifier.predict(embedding[np.newaxis, :], verbose=0)[0][0]
            )
            is_cough = prob > COUGH_THRESHOLD

            return {
                "probability": round(prob, 4),
                "is_cough":    is_cough,
                "label":       "cough" if is_cough else "not_cough",
                "confidence":  f"{max(prob, 1 - prob) * 100:.1f}%",
            }
        except Exception as exc:
            log.error(f"Cough detection error: {exc}")
            return {"error": str(exc)}


# ─── Voiceprint Engine ────────────────────────────────────────────────────────

class VoiceprintEngine:
    """
    Speaker verification with resemblyzer's pre-trained GE2E encoder.

    Enroll: store a d-vector embedding per name.
    Verify: cosine-similarity search against all enrolled profiles.

    Profiles are persisted to disk (voiceprint_profiles.npz) and survive restarts.
    Multiple enrollments for the same name are averaged for better accuracy.
    """

    def __init__(self):
        self._encoder = None
        self._profiles: dict[str, np.ndarray] = {}   # name → d-vector (256,)
        self._enrollment_counts: dict[str, int] = {}  # name → how many times enrolled
        self._load()
        self._load_profiles()

    def _load(self):
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            log.info("VoiceprintEngine ready (resemblyzer GE2E)  ✓")
        except Exception as exc:
            log.error(f"VoiceprintEngine init failed: {exc}")

    def _load_profiles(self):
        """Load persisted voiceprint profiles from disk."""
        if PROFILES_PATH.exists():
            try:
                data = np.load(str(PROFILES_PATH), allow_pickle=False)
                self._profiles = {k: data[k] for k in data.files if not k.endswith("__count")}
                self._enrollment_counts = {
                    k.replace("__count", ""): int(data[k])
                    for k in data.files if k.endswith("__count")
                }
                log.info(f"Loaded {len(self._profiles)} voiceprint profiles from disk")
            except Exception as exc:
                log.error(f"Failed to load voiceprint profiles: {exc}")

    def _save_profiles(self):
        """Persist voiceprint profiles to disk."""
        try:
            save_dict = {**self._profiles}
            for name, count in self._enrollment_counts.items():
                save_dict[f"{name}__count"] = np.array(count)
            np.savez(str(PROFILES_PATH), **save_dict)
        except Exception as exc:
            log.error(f"Failed to save voiceprint profiles: {exc}")

    @property
    def ready(self) -> bool:
        return self._encoder is not None

    def _embed(self, audio_bytes: bytes) -> np.ndarray:
        """Extract a 256-dim speaker embedding from raw audio bytes."""
        import librosa
        from resemblyzer import preprocess_wav

        try:
            wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            err = str(e)
            if "NoBackendError" in err or "ffmpeg" in err.lower() or "backend" in err.lower():
                raise RuntimeError(
                    "无法解码浏览器音频（WebM/Opus）。请安装 ffmpeg：brew install ffmpeg / "
                    "Cannot decode browser audio (WebM/Opus). Install ffmpeg: brew install ffmpeg"
                ) from e
            raise
        wav = preprocess_wav(wav.astype(np.float32), source_sr=SAMPLE_RATE)
        return self._encoder.embed_utterance(wav)   # (256,)

    def enroll(self, name: str, audio_bytes: bytes) -> dict:
        """
        Enroll a speaker by name.
        Multiple calls for the same name average the embeddings (improves accuracy).
        """
        if not self.ready:
            return {"error": "VoiceprintEngine not initialised"}
        try:
            new_emb = self._embed(audio_bytes)
            count = self._enrollment_counts.get(name, 0)
            if name in self._profiles:
                # Weighted average: give more weight to accumulated profile
                self._profiles[name] = (self._profiles[name] * count + new_emb) / (count + 1)
                self._enrollment_counts[name] = count + 1
                log.info(f"Updated voiceprint: {name} (sample #{count + 1})")
            else:
                self._profiles[name] = new_emb
                self._enrollment_counts[name] = 1
                log.info(f"Enrolled voiceprint: {name} (sample #1)")
            self._save_profiles()
            return {
                "success":          True,
                "name":             name,
                "samples":          self._enrollment_counts[name],
                "profiles_count":   len(self._profiles),
                "enrolled_names":   list(self._profiles.keys()),
            }
        except Exception as exc:
            log.error(f"Enroll error: {exc}")
            return {"error": str(exc)}

    def verify(self, audio_bytes: bytes) -> dict:
        """
        Identify the speaker in the audio.

        Returns:
            {
              "matched":    bool,
              "best_match": str | None,   # name with highest similarity
              "confidence": float,        # cosine similarity of best match
              "all_scores": dict,         # {name: similarity} for all profiles
              "threshold":  float,
            }
        """
        if not self.ready:
            return {"error": "VoiceprintEngine not initialised"}
        if not self._profiles:
            return {"error": "No voiceprints enrolled yet"}
        try:
            emb = self._embed(audio_bytes)
            scores: dict[str, float] = {}
            for name, profile in self._profiles.items():
                sim = float(
                    np.dot(emb, profile)
                    / (np.linalg.norm(emb) * np.linalg.norm(profile) + 1e-8)
                )
                scores[name] = round(sim, 4)

            best_name  = max(scores, key=scores.get)
            best_score = scores[best_name]
            matched    = best_score >= VOICEPRINT_THRESHOLD

            log.info(f"Verify: best={best_name} score={best_score:.3f} matched={matched}")
            return {
                "matched":    matched,
                "best_match": best_name if matched else None,
                "confidence": best_score,
                "all_scores": scores,
                "threshold":  VOICEPRINT_THRESHOLD,
            }
        except Exception as exc:
            log.error(f"Verify error: {exc}")
            return {"error": str(exc)}

    def list_profiles(self) -> list[str]:
        return list(self._profiles.keys())

    def delete_profile(self, name: str) -> bool:
        if name in self._profiles:
            del self._profiles[name]
            self._enrollment_counts.pop(name, None)
            self._save_profiles()
            log.info(f"Deleted voiceprint: {name}")
            return True
        return False

    def enrollment_count(self, name: str) -> int:
        """Return how many samples have been enrolled for this name."""
        return self._enrollment_counts.get(name, 0)
