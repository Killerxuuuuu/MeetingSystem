import os

import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):

    def _list_audio_backends():
        return ["soundfile"]

    torchaudio.list_audio_backends = _list_audio_backends

from pydub import AudioSegment

import config
from modules.asr import ASRHandler
from modules.diarizer import ManualDiarizer
from utils import merge_segments


def main():
    # Define input and output paths
    input_file = "data/ami_test_meeting.wav"
    output_file = "data/output_result.txt"

    # Validation check
    if not os.path.exists(input_file):
        print(
            f"Error: File {input_file} not found. Please place a .wav file in the 'data' folder."
        )
        return

    # --- Phase 1: Speaker Diarization (Feature Extraction + Clustering) ---
    print("\n>>> Phase 1: Initializing Manual Speaker Diarization Engine...")
    diarizer = ManualDiarizer()

    # Run the sliding window and spectral clustering algorithm
    raw_segments = diarizer.run(input_file, num_speakers=config.NUM_SPEAKERS)
    print(f"    Raw sliding window segments generated: {len(raw_segments)}")

    # --- Phase 2: Segment Merging (Algorithmic Logic) ---
    print("\n>>> Phase 2: Executing Segment Merging Algorithm...")
    # Merge overlapping or adjacent windows belonging to the same cluster
    merged_segments = merge_segments(raw_segments)
    print(f"    Valid conversational turns after merging: {len(merged_segments)}")

    # --- Phase 3: Automatic Speech Recognition (ASR) ---
    print("\n>>> Phase 3: Executing Segmented Speech Recognition...")
    asr_handler = ASRHandler()

    # Load source audio for physical slicing using pydub
    source_audio = AudioSegment.from_wav(input_file)

    final_results = []

    for i, seg in enumerate(merged_segments):
        # Convert seconds to milliseconds for pydub
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)

        # Physically slice the audio chunk
        chunk_audio = source_audio[start_ms:end_ms]

        # Export to a temporary file for Whisper to read
        temp_filename = f"data/temp_{i}.wav"
        chunk_audio.export(temp_filename, format="wav")

        # Transcribe the temp file
        text = asr_handler.transcribe_file(temp_filename)

        # Format the output if text is not empty
        if text.strip():
            speaker_name = f"Speaker {seg['speaker']}"
            time_str = f"[{seg['start']:.2f}s -> {seg['end']:.2f}s]"
            line = f"{time_str} {speaker_name}: {text}"

            print(line)
            final_results.append(line)

        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    # --- Phase 4: Saving Results ---
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(final_results))

    print(f"\nProcessing Complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
