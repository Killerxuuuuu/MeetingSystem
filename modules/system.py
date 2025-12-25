import signal
import sys
import time

import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):

    def _list_audio_backends():
        return ["soundfile"]

    torchaudio.list_audio_backends = _list_audio_backends

from modules.stream import StreamingAudioCapture, StreamingMeetingProcessor


class StreamingMeetingSystem:
    def __init__(self, output_file="data/streaming_output.txt"):
        self.audio_capture = StreamingAudioCapture()
        self.processor = StreamingMeetingProcessor()
        self.running = False
        self.output_file = output_file
        self.file_handle = None

    def start(self):
        """Start the streaming meeting system"""
        print("Starting streaming meeting system...")
        print("Press Ctrl+C to stop")

        # Open the output file for writing
        self.file_handle = open(self.output_file, "w", encoding="utf-8")

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        # Start audio capture
        self.audio_capture.start_stream()
        self.running = True

        print("🎤 Listening... Speak into your microphone")

        # Main processing loop
        while self.running:
            # Get audio chunk
            chunk_data = self.audio_capture.get_audio_chunk()

            if chunk_data:
                audio_data, timestamp = chunk_data
                self.processor.add_audio_chunk(audio_data, timestamp)

            # Process buffer if needed
            if self.processor.should_process():
                print("🔄 Processing audio buffer...")
                results = self.processor.process_buffer()

                # Display and write results
                for result in results:
                    timestamp_str = time.strftime(
                        "%H:%M:%S", time.localtime(result["timestamp"])
                    )
                    line = f"[{timestamp_str}] {result['speaker']}: {result['text']}"
                    print(line)
                    if self.file_handle:
                        self.file_handle.write(line + "\n")
                        self.file_handle.flush()

            # Small sleep to prevent CPU spinning
            time.sleep(0.1)

    def stop(self):
        """Stop the streaming system"""
        print("\n🛑 Stopping streaming system...")
        self.running = False
        self.audio_capture.stop_stream()
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.stop()
        sys.exit(0)
