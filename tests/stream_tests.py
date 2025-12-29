import time

import numpy as np
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):

    def _list_audio_backends():
        return ["soundfile"]

    torchaudio.list_audio_backends = _list_audio_backends

from modules.system import StreamingMeetingSystem


class SimulatedAudioCapture:
    def __init__(self, input_file, chunk_duration=10.0):
        self.input_file = input_file
        self.chunk_duration = chunk_duration  # seconds
        self.waveform, self.sample_rate = torchaudio.load(input_file)
        self.audio_data = self.waveform.squeeze().numpy()
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.pointer = 0

    def start_stream(self):
        pass  # No-op for simulation

    def stop_stream(self):
        pass  # No-op for simulation

    def get_audio_chunk(self):
        if self.pointer >= len(self.audio_data):
            return None
        chunk = self.audio_data[self.pointer : self.pointer + self.chunk_samples]
        self.pointer += self.chunk_samples
        # Simulate real-time by sleeping
        time.sleep(self.chunk_duration)
        # Convert to int16 like microphone input
        chunk = (chunk * 32767).astype(np.int16)
        return chunk, time.time()


def main():
    input_file = "data/ami_test_meeting.wav"
    output_file = "data/streaming_output.txt"
    # Patch the system to use simulated audio capture
    system = StreamingMeetingSystem(output_file=output_file)
    system.audio_capture = SimulatedAudioCapture(input_file, chunk_duration=10.0)
    system.start()


if __name__ == "__main__":
    main()
