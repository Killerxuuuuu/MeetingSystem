from datasets import load_dataset
import soundfile as sf
import os

os.makedirs("data", exist_ok=True)

dataset = load_dataset("diarizers-community/ami", "ihm", split="test", streaming=True)

sample = next(iter(dataset))
audio_array = sample["audio"]["array"]
sampling_rate = sample["audio"]["sampling_rate"]

output_path = "data/ami_test_meeting.wav"
sf.write(output_path, audio_array, sampling_rate)
