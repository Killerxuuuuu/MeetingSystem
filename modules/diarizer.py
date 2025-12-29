import numpy as np
import torch
import torchaudio
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from speechbrain.inference.classifiers import EncoderClassifier
from tqdm import tqdm

from modules import config


class ManualDiarizer:
    def __init__(self):
        print("[Init] Loading ECAPA-TDNN Speaker Embedding Model...")
        self.device = config.DEVICE

        # Load the pre-trained encoder from SpeechBrain
        # We use this to extract d-vectors/x-vectors, not the full pipeline
        self.encoder = EncoderClassifier.from_hparams(
            source=config.EMBEDDING_MODEL, run_opts={"device": self.device}
        )

    def _extract_embedding(self, wav_tensor):
        """
        Convert a waveform tensor into a 192-dimensional feature vector.
        """
        with torch.no_grad():
            # SpeechBrain expects input normalized, though not strictly required, it's good practice.
            # encode_batch output shape: [batch, 1, 192]
            embeddings = self.encoder.encode_batch(wav_tensor)

            # Squeeze dimensions to get a flat vector -> [192]
            return embeddings.squeeze().cpu().numpy()

    def run(self, audio_path, num_speakers=config.NUM_SPEAKERS):
        """
        Execute the full manual diarization pipeline:
        1. Load Audio -> 2. Sliding Window -> 3. Feature Extraction -> 4. Clustering
        """

        # 1. Load Audio
        # sig shape: [channels, time], fs: sample rate
        sig, fs = torchaudio.load(audio_path)

        # Convert stereo to mono if necessary
        if sig.shape[0] > 1:
            sig = torch.mean(sig, dim=0, keepdim=True)

        # 2. Prepare Sliding Window Parameters
        window_samples = int(config.WINDOW_SIZE * fs)
        step_samples = int(config.STEP_SIZE * fs)
        total_samples = sig.shape[1]

        embeddings = []
        timestamps = []  # Store time range for each window: [start, end]

        print(
            f"[Step 1] Starting Sliding Window Feature Extraction (Total Duration: {total_samples / fs:.2f}s)..."
        )

        # 3. Sliding Window Loop
        # Include final window by adding +1 in the range upper bound
        for start in tqdm(
            range(0, max(0, total_samples - window_samples) + 1, step_samples)
        ):
            end = start + window_samples
            if end > total_samples:
                # pad with zeros if needed (keeps consistent window length)
                chunk = torch.zeros((1, window_samples), device=self.device)
                available = sig[:, start:total_samples].to(self.device)
                chunk[:, : available.shape[1]] = available
            else:
                # Extract the chunk and move to device
                chunk = sig[:, start:end].to(self.device)

            # Extract speaker vector (embedding)
            try:
                emb = self._extract_embedding(chunk)
            except Exception as ex:
                # If encoder failed on this window, skip it and log
                print(f"[Warning] encoder failed for window {start}:{end} -> {ex}")
                continue

            embeddings.append(emb)
            timestamps.append((start / fs, min(end, total_samples) / fs))

        if not embeddings:
            raise ValueError(
                "Audio is too short to extract features or all windows failed."
            )

        # Convert list to numpy array: [N_windows, D]
        X = np.array(embeddings)

        # --- SANITIZE embeddings: remove NaN/Inf or zero-norm windows ---
        nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
        norms = np.linalg.norm(X, axis=1)
        zero_mask = norms < 1e-8
        bad_mask = nan_mask | zero_mask

        if bad_mask.any():
            removed = int(bad_mask.sum())
            print(f"[Warning] Removing {removed} windows with NaN/Inf/zero embeddings.")
            keep_idx = np.where(~bad_mask)[0]
            X = X[keep_idx]
            timestamps = [timestamps[i] for i in keep_idx]

        if X.shape[0] == 0:
            raise ValueError("All sliding windows produced invalid embeddings.")

        # If after filtering there are fewer windows than speakers, fall back.
        if X.shape[0] < num_speakers:
            print(
                f"[Warning] only {X.shape[0]} valid windows for {num_speakers} speakers; "
                "falling back to single-speaker labeling for these windows."
            )
            labels = np.zeros(X.shape[0], dtype=int)
        else:
            # Normalize embeddings and compute a stable precomputed cosine affinity matrix
            Xn = normalize(X, axis=1)
            affinity = np.dot(Xn, Xn.T)

            # Sanitize affinity matrix numerically
            affinity = np.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(affinity, 1.0)

            print(
                f"[Step 2] Performing Spectral Clustering (Target Speakers: {num_speakers})..."
            )
            cluster_model = SpectralClustering(
                n_clusters=num_speakers,
                affinity="precomputed",  # we provide a stabilized affinity matrix
                assign_labels="kmeans",
                random_state=42,
            )
            labels = cluster_model.fit_predict(affinity)

        # 5. Map cluster labels back to timestamps
        # Format: list of {'start': 0.0, 'end': 2.0, 'speaker': 0}
        raw_segments = []
        for i, label in enumerate(labels):
            raw_segments.append(
                {
                    "start": timestamps[i][0],
                    "end": timestamps[i][1],
                    "speaker": int(label),
                }
            )

        return raw_segments
