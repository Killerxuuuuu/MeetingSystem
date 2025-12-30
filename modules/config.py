# config.py

# Algorithm Hyperparameters
NUM_SPEAKERS = None  # Target number of speakers to distinguish (Set to None for auto-detection)
WINDOW_SIZE = 2.0  # Sliding window size (in seconds) for feature extraction
STEP_SIZE = 1.0  # Step size (in seconds), 50% overlap

# Model Paths / Settings
EMBEDDING_MODEL = (
    "speechbrain/spkrec-ecapa-voxceleb"  # Pre-trained encoder for speaker vectors
)
ASR_MODEL_SIZE = "medium"  # Whisper model size: tiny, base, small, medium, large-v3
DEVICE = "cuda"  # Use "cuda" if GPU is available, otherwise "cpu"
