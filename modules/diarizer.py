import numpy as np
import torch
import torchaudio
from sklearn.cluster import SpectralClustering
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
        # Iterate through the audio file with a step size
        for start in tqdm(range(0, total_samples - window_samples, step_samples)):
            end = start + window_samples

            # Extract the chunk and move to device
            chunk = sig[:, start:end].to(self.device)

            # Extract speaker vector (embedding)
            emb = self._extract_embedding(chunk)
            embeddings.append(emb)
            timestamps.append((start / fs, end / fs))

        if not embeddings:
            raise ValueError("Audio is too short to extract features.")

        # Convert list to numpy array: [N_windows, 192]
        X = np.array(embeddings)

        # 4. Perform Spectral Clustering
        # We use Spectral Clustering because it works well with Cosine Similarity (affinity)
        # This is the core logic separating speakers based on vector similarity.
        print(
            f"[Step 2] Performing Spectral Clustering (Target Speakers: {num_speakers})..."
        )
        cluster_model = SpectralClustering(
            n_clusters=num_speakers,
            affinity="cosine",  # Cosine similarity is standard for speaker embeddings
            assign_labels="kmeans",
            random_state=42,
        )
        labels = cluster_model.fit_predict(X)

        # 5. Map cluster labels back to timestamps
        # Format: list of {'start': 0.0, 'end': 2.0, 'speaker': 0}
        raw_segments = []
        for i, label in enumerate(labels):
            raw_segments.append(
                {"start": timestamps[i][0], "end": timestamps[i][1], "speaker": label}
            )

        return raw_segments
