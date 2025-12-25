import queue
import time
from collections import deque

import numpy as np
import pyaudio
import torch
import torchaudio
from pydub import AudioSegment

from .asr import ASRHandler
from .diarizer import ManualDiarizer


class StreamingAudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        format: int = pyaudio.paInt16,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format

        self.audio_queue = queue.Queue()
        self.is_streaming = False
        self.stream = None
        self.audio = None

    def start_stream(self):
        """Start capturing audio from microphone"""
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )
        self.is_streaming = True
        self.stream.start_stream()

    def stop_stream(self):
        """Stop audio capture"""
        self.is_streaming = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_streaming:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_queue.put((audio_data, time.time()))
        return (None, pyaudio.paContinue)

    def get_audio_chunk(self) -> tuple | None:
        """Get next audio chunk from queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None


class StreamingMeetingProcessor:
    def __init__(
        self,
        buffer_duration: float = 60.0,  # 60 seconds of audio buffer
        processing_interval: float = 10.0,  # Process every 10 seconds
        sample_rate: int = 16000,
    ):
        self.buffer_duration = buffer_duration
        self.processing_interval = processing_interval
        self.sample_rate = sample_rate

        # Circular buffer for audio chunks
        self.audio_buffer = deque()
        self.buffer_timestamps = deque()

        # Initialize components
        self.diarizer = ManualDiarizer()
        self.asr_handler = ASRHandler()

        # State tracking
        self.last_process_time = 0
        self.speaker_history = {}  # Track speaker embeddings over time
        self.current_speaker_id = 0

    def add_audio_chunk(self, audio_data: np.ndarray, timestamp: float):
        """Add new audio chunk to buffer"""
        self.audio_buffer.append(audio_data)
        self.buffer_timestamps.append(timestamp)

        # Remove old data beyond buffer duration
        current_time = timestamp
        while (
            self.buffer_timestamps
            and current_time - self.buffer_timestamps[0] > self.buffer_duration
        ):
            self.audio_buffer.popleft()
            self.buffer_timestamps.popleft()

    def should_process(self) -> bool:
        """Check if it's time to process the buffer"""
        current_time = time.time()
        return (current_time - self.last_process_time) >= self.processing_interval

    def process_buffer(self) -> list[dict]:
        """Process current buffer and return results"""
        if not self.audio_buffer:
            return []

        # Concatenate buffer into single audio array
        full_audio = np.concatenate(list(self.audio_buffer))

        # Convert to torch tensor and save as temporary file
        audio_tensor = torch.from_numpy(full_audio).float().unsqueeze(0)
        audio_tensor = audio_tensor / 32768.0  # Normalize int16 to float

        # Save to temporary file for processing
        temp_path = "data/temp_streaming.wav"
        torchaudio.save(temp_path, audio_tensor, self.sample_rate)

        try:
            # Run diarization on buffer
            raw_segments = self.diarizer.run(temp_path, num_speakers=2)

            # Filter to recent segments only (last processing_interval seconds)
            current_time = time.time()
            recent_segments = []
            for segment in raw_segments:
                # Convert relative time to absolute time
                segment_time = self.buffer_timestamps[0] + segment["start"]
                if current_time - segment_time <= self.processing_interval * 1.5:
                    recent_segments.append(
                        {
                            "start": segment_time,
                            "end": self.buffer_timestamps[0] + segment["end"],
                            "speaker": segment["speaker"],
                            "relative_start": segment["start"],
                            "relative_end": segment["end"],
                        }
                    )

            # Transcribe recent segments
            results = []
            source_audio = AudioSegment.from_wav(temp_path)

            for segment in recent_segments:
                # Extract audio chunk
                start_ms = int(segment["relative_start"] * 1000)
                end_ms = int(segment["relative_end"] * 1000)

                # --- Skip very short segments (< 1.0s) ---
                if (end_ms - start_ms) < 1000:
                    continue

                chunk_audio = source_audio[start_ms:end_ms]

                # --- Skip silence (Energy Threshold) ---
                # RMS (Root Mean Square) amplitude is a measure of loudness.
                # Adjust this threshold if needed (e.g., 100-500).
                if chunk_audio.rms < 200:
                    continue

                # Transcribe
                chunk_path = "data/temp_chunk.wav"
                chunk_audio.export(chunk_path, format="wav")
                text = self.asr_handler.transcribe_file(chunk_path)

                # ---  Filter Hallucinations ---
                clean_text = text.strip()
                if not clean_text:
                    continue

                # Common Whisper hallucinations on silence
                hallucinations = [
                    "thanks for watching",
                    "thank you",
                    "subtitles by",
                    "amara.org",
                    "copyright",
                    "all rights reserved",
                    "you",
                    "nope",
                ]

                # Check if the text is just a hallucination (case-insensitive)
                if (
                    any(h in clean_text.lower() for h in hallucinations)
                    and len(clean_text.split()) < 5
                ):
                    continue

                results.append(
                    {
                        "timestamp": segment["start"],
                        "speaker": f"Speaker {segment['speaker']}",
                        "text": clean_text,
                        "duration": segment["end"] - segment["start"],
                    }
                )

            self.last_process_time = time.time()
            return results

        except Exception as e:
            print(f"Processing error: {e}")
            return []
