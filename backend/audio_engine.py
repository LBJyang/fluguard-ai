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

# Trained classifier head lives alongside the backend code in the models/ subfolder
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

    def _load_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decode audio bytes to 16 kHz mono float32 array using librosa."""
        import librosa
        audio_file = io.BytesIO(audio_bytes)
        waveform, _ = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
        # Normalise to [-1, 1]
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
        return waveform

    def detect(self, audio_bytes: bytes) -> dict:
        """
        Detect cough in audio bytes.
        Returns: { probability, is_cough, label, confidence }
        """
        import tensorflow as tf
        try:
            waveform = self._load_audio(audio_bytes)
            waveform_tensor = tf.cast(waveform, tf.float32)

            # YAMNet: returns (scores, embeddings, spectrogram)
            _, embeddings, _ = self._yamnet(waveform_tensor)
            # Average frame embeddings → single 1024-dim vector
            mean_embedding = tf.reduce_mean(embeddings, axis=0, keepdims=True)

            # Classifier head
            prob_tensor = self._classifier(mean_embedding)
            probability = float(prob_tensor.numpy()[0][0])

            is_cough = probability >= COUGH_THRESHOLD
            return {
                "probability": round(probability, 4),
                "is_cough": is_cough,
                "label": "cough" if is_cough else "no_cough",
                "confidence": round(abs(probability - 0.5) * 2, 4),
                "threshold": COUGH_THRESHOLD,
            }
        except Exception as exc:
            return {"error": str(exc)}


# ─── Voiceprint Engine ────────────────────────────────────────────────────────

class VoiceprintEngine:
    """
    Speaker identification via resemblyzer GE2E encoder.

    Profiles are persisted to disk as an .npz file so they survive server restarts.
    """

    def __init__(self, profiles_path: Path = PROFILES_PATH):
        self._encoder = None
        self._profiles: dict[str, np.ndarray] = {}
        self._profiles_path = profiles_path
        self._load()

    def _load(self):
        try:
            from resemblyzer import VoiceEncoder
            log.info("Loading resemblyzer GE2E encoder...")
            self._encoder = VoiceEncoder()
            log.info("VoiceprintEngine ready  ✓")
        except Exception as exc:
            log.error(f"VoiceprintEngine init failed: {exc}")
            return

        # Load persisted profiles
        if self._profiles_path.exists():
            try:
                data = np.load(str(self._profiles_path), allow_pickle=True)
                self._profiles = {k: data[k] for k in data.files}
                log.info(f"Loaded {len(self._profiles)} voiceprint profiles from disk")
            except Exception as exc:
                log.warning(f"Could not load voiceprint profiles: {exc}")

    @property
    def ready(self) -> bool:
        return self._encoder is not None

    def _save_profiles(self):
        """Persist profiles to disk."""
        try:
            np.savez(str(self._profiles_path), **self._profiles)
        except Exception as exc:
            log.warning(f"Could not save voiceprint profiles: {exc}")

    def _audio_to_embedding(self, audio_bytes: bytes) -> np.ndarray:
        """Decode audio and produce a 256-dim d-vector."""
        import librosa
        from resemblyzer import preprocess_wav
        audio_file = io.BytesIO(audio_bytes)
        waveform, _ = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
        wav = preprocess_wav(waveform, source_sr=SAMPLE_RATE)
        return self._encoder.embed_utterance(wav)

    def list_profiles(self) -> list[str]:
        return list(self._profiles.keys())

    def enroll(self, name: str, audio_bytes: bytes) -> dict:
        """Enroll a speaker. Returns {success, name, embedding_dim}."""
        try:
            embedding = self._audio_to_embedding(audio_bytes)
            self._profiles[name] = embedding
            self._save_profiles()
            return {"success": True, "name": name, "embedding_dim": len(embedding)}
        except Exception as exc:
            return {"error": str(exc)}

    def verify(self, audio_bytes: bytes) -> dict:
        """
        Identify who is speaking.
        Returns: { matched, best_match, confidence, all_scores, threshold }
        """
        if not self._profiles:
            return {"matched": False, "best_match": None, "confidence": 0.0,
                    "all_scores": {}, "threshold": VOICEPRINT_THRESHOLD,
                    "note": "No profiles enrolled"}
        try:
            query_emb = self._audio_to_embedding(audio_bytes)
            scores = {}
            for name, profile_emb in self._profiles.items():
                cosine_sim = float(np.dot(query_emb, profile_emb) /
                                   (np.linalg.norm(query_emb) * np.linalg.norm(profile_emb)))
                scores[name] = round(cosine_sim, 4)

            best_match = max(scores, key=scores.__getitem__)
            best_score = scores[best_match]
            matched = best_score >= VOICEPRINT_THRESHOLD

            return {
                "matched": matched,
                "best_match": best_match if matched else None,
                "confidence": best_score,
                "all_scores": scores,
                "threshold": VOICEPRINT_THRESHOLD,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def delete_profile(self, name: str) -> bool:
        if name not in self._profiles:
            return False
        del self._profiles[name]
        self._save_profiles()
        return True
