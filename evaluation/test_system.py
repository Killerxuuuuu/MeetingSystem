"""
Test script to run the MeetingSystem on the existing test data
and generate some basic performance metrics
"""

import os
import time

import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):

    def _list_audio_backends():
        return ["soundfile"]

    torchaudio.list_audio_backends = _list_audio_backends

from evaluation.evaluate_performance import PerformanceEvaluator


def test_system_with_existing_data():
    """Test the system with existing data and collect basic metrics"""
    print("Testing MeetingSystem with existing data...")

    # Check if test data exists
    test_audio_path = "data/ami_test_meeting.wav"
    if not os.path.exists(test_audio_path):
        print(f"Test audio not found: {test_audio_path}")
        print("Please run download.py first or provide your own audio file")
        return

    print(f"Found test audio: {test_audio_path}")

    try:
        audio_tensor, sample_rate = torchaudio.load(test_audio_path)
        duration = audio_tensor.shape[1] / sample_rate
        print(f"Audio duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"Could not load audio file: {e}")
        return

    # Initialize performance evaluator
    evaluator = PerformanceEvaluator()

    # Run performance evaluation on the test file
    print("\nRunning performance evaluation...")
    results = evaluator.run_comprehensive_evaluation(test_audio_path)

    # Generate reports
    evaluator.generate_report("test_evaluation_report.json")
    evaluator.plot_performance("test_performance_plots.png")

    print("\nTest completed successfully!")
    print("Check the generated reports for detailed metrics.")


def run_basic_functionality_test():
    """Run a basic functionality test"""
    print("Running basic functionality test...")

    # Test individual components
    from modules.asr import ASRHandler
    from modules.diarizer import ManualDiarizer

    # Test diarizer
    test_audio = "data/ami_test_meeting.wav"
    if os.path.exists(test_audio):
        print("Testing diarizer component...")
        try:
            diarizer = ManualDiarizer()
            start_time = time.time()
            segments = diarizer.run(test_audio, num_speakers=2)
            diarization_time = time.time() - start_time
            print(f"Diarization completed in {diarization_time:.2f}s")
            print(f"Generated {len(segments)} segments")
        except Exception as e:
            print(f"Diarization test failed: {e}")

        # Test ASR
        print("Testing ASR component...")
        try:
            asr_handler = ASRHandler()
            start_time = time.time()
            text = asr_handler.transcribe_file(test_audio)
            asr_time = time.time() - start_time
            print(f"ASR completed in {asr_time:.2f}s")
            print(f"Transcribed text length: {len(text)} characters")
        except Exception as e:
            print(f"ASR test failed: {e}")
    else:
        print(f"Test audio file {test_audio} not found. Run download.py first.")


if __name__ == "__main__":
    print("MeetingSystem Test Suite")
    print("=" * 40)

    # Run basic functionality test
    run_basic_functionality_test()

    print("\n" + "=" * 40)

    # Run performance evaluation
    test_system_with_existing_data()

    print("\nAll tests completed!")
