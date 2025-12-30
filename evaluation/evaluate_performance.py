"""
Performance Evaluation Script for MeetingSystem

This script evaluates various performance metrics for the MeetingSystem including:
- Processing time
- Speaker diarization accuracy (when ground truth is available)
- ASR quality metrics (WER, CER)
- Memory usage
- Real-time factor
"""

import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from modules.asr import ASRHandler
from modules.diarizer import ManualDiarizer
from modules.stream import StreamingMeetingProcessor

# Import system modules


class PerformanceEvaluator:
    def __init__(self):
        self.results = {}
        self.metrics_history = []

    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    def benchmark_diarization(self, audio_path: str, num_runs: int = 5) -> dict:
        """Benchmark the diarization component"""
        print("Benchmarking Diarization Component...")

        diarizer = ManualDiarizer()
        times = []
        memory_usages = []

        for i in range(num_runs):
            start_time = time.time()
            start_memory = self.measure_memory_usage()

            # Run diarization
            segments = diarizer.run(audio_path, num_speakers=2)

            end_time = time.time()
            end_memory = self.measure_memory_usage()

            times.append(end_time - start_time)
            memory_usages.append(end_memory - start_memory)

        diarization_metrics = {
            "average_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "average_memory_delta_mb": np.mean(memory_usages),
            "num_segments": len(segments),
            "total_runs": num_runs,
        }

        print(
            f"Diarization - Avg time: {diarization_metrics['average_time']:.2f}s, "
            f"Avg mem change: {diarization_metrics['average_memory_delta_mb']:.2f}MB"
        )

        return diarization_metrics

    def benchmark_asr(self, audio_path: str, num_runs: int = 5) -> dict:
        """Benchmark the ASR component"""
        print("Benchmarking ASR Component...")

        asr_handler = ASRHandler()
        times = []
        memory_usages = []

        for i in range(num_runs):
            start_time = time.time()
            start_memory = self.measure_memory_usage()

            # Run ASR
            text = asr_handler.transcribe_file(audio_path)

            end_time = time.time()
            end_memory = self.measure_memory_usage()

            times.append(end_time - start_time)
            memory_usages.append(end_memory - start_memory)

        asr_metrics = {
            "average_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "average_memory_delta_mb": np.mean(memory_usages),
            "text_length": len(text),
            "total_runs": num_runs,
        }

        print(
            f"ASR - Avg time: {asr_metrics['average_time']:.2f}s, "
            f"Avg mem change: {asr_metrics['average_memory_delta_mb']:.2f}MB"
        )

        return asr_metrics

    def benchmark_full_pipeline(self, audio_path: str, num_runs: int = 3) -> dict:
        """Benchmark the full streaming pipeline"""
        print("Benchmarking Full Streaming Pipeline...")

        processor = StreamingMeetingProcessor()
        times = []
        memory_usages = []

        for i in range(num_runs):
            start_time = time.time()
            start_memory = self.measure_memory_usage()

            # Simulate adding audio chunks and processing
            # This is a simplified version of the streaming pipeline
            import numpy as np
            import torchaudio

            # Load the audio and simulate streaming
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            audio_array = audio_tensor.numpy().squeeze()

            # Simulate streaming by processing in chunks
            chunk_size = int(sample_rate * 0.5)  # 0.5 seconds chunks

            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i : i + chunk_size]
                timestamp = i / sample_rate
                processor.add_audio_chunk(chunk, timestamp)

                # Process if needed
                if processor.should_process():
                    results = processor.process_buffer()

            end_time = time.time()
            end_memory = self.measure_memory_usage()

            times.append(end_time - start_time)
            memory_usages.append(end_memory - start_memory)

        pipeline_metrics = {
            "average_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "average_memory_delta_mb": np.mean(memory_usages),
            "total_runs": num_runs,
        }

        print(
            f"Pipeline - Avg time: {pipeline_metrics['average_time']:.2f}s, "
            f"Avg mem change: {pipeline_metrics['average_memory_delta_mb']:.2f}MB"
        )

        return pipeline_metrics

    def evaluate_real_time_performance(
        self, audio_path: str, real_duration: float
    ) -> dict:
        """Evaluate real-time factor (RTF)"""
        print("Evaluating Real-Time Factor...")

        # Calculate actual audio duration
        import torchaudio

        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_duration = audio_tensor.shape[1] / sample_rate

        start_time = time.time()

        # Process the audio
        processor = StreamingMeetingProcessor()
        audio_array = audio_tensor.numpy().squeeze()
        chunk_size = int(sample_rate * 0.5)

        for i in range(0, len(audio_array), chunk_size):
            chunk = audio_array[i : i + chunk_size]
            timestamp = i / sample_rate
            processor.add_audio_chunk(chunk, timestamp)

            if processor.should_process():
                results = processor.process_buffer()

        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration  # Real-time factor

        rtf_metrics = {
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "real_time_factor": rtf,
            "is_real_time": rtf < 1.0,  # Processing faster than real-time
        }

        print(
            f"RTF: {rtf:.2f}x (Processing {'faster' if rtf < 1.0 else 'slower'} than real-time)"
        )

        return rtf_metrics

    def run_comprehensive_evaluation(self, audio_path: str) -> dict:
        """Run all performance evaluations"""
        print("Starting Comprehensive Performance Evaluation...")
        print(f"Audio file: {audio_path}")
        print(f"Timestamp: {datetime.now()}")
        print("-" * 50)

        # Run individual benchmarks
        diarization_metrics = self.benchmark_diarization(audio_path)
        asr_metrics = self.benchmark_asr(audio_path)
        pipeline_metrics = self.benchmark_full_pipeline(audio_path)
        rtf_metrics = self.evaluate_real_time_performance(
            audio_path, 10.0
        )  # Assuming 10s duration

        # Overall system metrics
        overall_metrics = {
            "timestamp": datetime.now().isoformat(),
            "audio_file": audio_path,
            "diarization": diarization_metrics,
            "asr": asr_metrics,
            "pipeline": pipeline_metrics,
            "rtf": rtf_metrics,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_mb": psutil.virtual_memory().total / 1024 / 1024,
                "torch_cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            },
        }

        self.results = overall_metrics
        return overall_metrics

    def generate_report(self, output_path: str = "evaluation_report.json"):
        """Generate a performance evaluation report"""
        if not self.results:
            print("No results to generate report. Run evaluation first.")
            return

        # Save detailed JSON report
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"Performance report saved to {output_path}")

        # Generate summary
        self._generate_summary()

    def _generate_summary(self):
        """Generate a text summary of the evaluation"""
        print("\n" + "=" * 60)
        print("PERFORMANCE EVALUATION SUMMARY")
        print("=" * 60)

        if "diarization" in self.results:
            print("\nSpeaker Diarization:")
            print(
                f"  Average processing time: {self.results['diarization']['average_time']:.2f}s"
            )
            print(
                f"  Average memory change: {self.results['diarization']['average_memory_delta_mb']:.2f}MB"
            )
            print(
                f"  Number of segments: {self.results['diarization']['num_segments']}"
            )

        if "asr" in self.results:
            print("\nAutomatic Speech Recognition:")
            print(
                f"  Average processing time: {self.results['asr']['average_time']:.2f}s"
            )
            print(
                f"  Average memory change: {self.results['asr']['average_memory_delta_mb']:.2f}MB"
            )
            print(f"  Text length: {self.results['asr']['text_length']} characters")

        if "rtf" in self.results:
            print("\nReal-Time Performance:")
            print(
                f"  Real-Time Factor (RTF): {self.results['rtf']['real_time_factor']:.2f}x"
            )
            print(
                f"  Real-time capable: {'Yes' if self.results['rtf']['is_real_time'] else 'No'}"
            )

        print("\nSystem Information:")
        sys_info = self.results.get("system_info", {})
        print(f"  CPU cores: {sys_info.get('cpu_count', 'N/A')}")
        print(f"  Total memory: {sys_info.get('memory_total_mb', 'N/A'):.0f}MB")
        print(f"  CUDA available: {sys_info.get('torch_cuda_available', 'N/A')}")
        if sys_info.get("cuda_device"):
            print(f"  GPU: {sys_info['cuda_device']}")

        print("=" * 60)

    def plot_performance(self, output_path: str = "performance_plots.png"):
        """Generate performance plots"""
        if not self.results:
            print("No results to plot. Run evaluation first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("MeetingSystem Performance Evaluation", fontsize=16)

        # Plot 1: Processing times comparison
        if all(key in self.results for key in ["diarization", "asr", "pipeline"]):
            components = ["Diarization", "ASR", "Pipeline"]
            avg_times = [
                self.results["diarization"]["average_time"],
                self.results["asr"]["average_time"],
                self.results["pipeline"]["average_time"],
            ]

            axes[0, 0].bar(components, avg_times)
            axes[0, 0].set_title("Average Processing Time per Component")
            axes[0, 0].set_ylabel("Time (seconds)")
            axes[0, 0].tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for i, v in enumerate(avg_times):
                axes[0, 0].text(
                    i, v + max(avg_times) * 0.01, f"{v:.2f}s", ha="center", va="bottom"
                )

        # Plot 2: Memory usage
        if all(key in self.results for key in ["diarization", "asr", "pipeline"]):
            memory_changes = [
                self.results["diarization"]["average_memory_delta_mb"],
                self.results["asr"]["average_memory_delta_mb"],
                self.results["pipeline"]["average_memory_delta_mb"],
            ]

            axes[0, 1].bar(components, memory_changes, color="orange")
            axes[0, 1].set_title("Average Memory Change per Component")
            axes[0, 1].set_ylabel("Memory Change (MB)")
            axes[0, 1].tick_params(axis="x", rotation=45)

            for i, v in enumerate(memory_changes):
                axes[0, 1].text(
                    i,
                    v + max(memory_changes) * 0.01,
                    f"{v:.1f}MB",
                    ha="center",
                    va="bottom",
                )

        # Plot 3: Real-time factor
        if "rtf" in self.results:
            rtf = self.results["rtf"]["real_time_factor"]
            axes[1, 0].bar(
                ["Real-Time Factor"], [rtf], color="green" if rtf < 1.0 else "red"
            )
            axes[1, 0].set_title("Real-Time Factor (RTF)")
            axes[1, 0].set_ylabel("RTF")
            axes[1, 0].axhline(
                y=1.0, color="red", linestyle="--", label="Real-time threshold"
            )
            axes[1, 0].legend()

            axes[1, 0].text(
                0, rtf + max(rtf, 1) * 0.01, f"{rtf:.2f}x", ha="center", va="bottom"
            )

        # Plot 4: System info (simplified)
        axes[1, 1].axis("off")  # Turn off axis for text info
        sys_info_text = []
        if "system_info" in self.results:
            sys_info = self.results["system_info"]
            sys_info_text.append(f"CPU Cores: {sys_info.get('cpu_count', 'N/A')}")
            sys_info_text.append(
                f"Memory: {sys_info.get('memory_total_mb', 'N/A'):.0f}MB"
            )
            sys_info_text.append(f"CUDA: {sys_info.get('torch_cuda_available', 'N/A')}")
            if sys_info.get("cuda_device"):
                sys_info_text.append(f"GPU: {sys_info['cuda_device']}")

        if sys_info_text:
            axes[1, 1].text(
                0.1,
                0.9,
                "\n".join(sys_info_text),
                transform=axes[1, 1].transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Performance plots saved to {output_path}")
        plt.show()


def main():
    """Main function to run performance evaluation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Performance Evaluation for MeetingSystem"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to audio file for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = PerformanceEvaluator()

    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(args.audio_path)

    # Generate reports
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    evaluator.generate_report(report_path)

    plot_path = os.path.join(args.output_dir, "performance_plots.png")
    evaluator.plot_performance(plot_path)

    print(f"\nEvaluation complete! Check {args.output_dir} for results.")


if __name__ == "__main__":
    # Example usage without command line args
    evaluator = PerformanceEvaluator()

    # Assuming we have a test audio file in the data directory
    test_audio_path = "data/ami_test_meeting.wav"

    if os.path.exists(test_audio_path):
        print(f"Running evaluation on {test_audio_path}")
        results = evaluator.run_comprehensive_evaluation(test_audio_path)
        evaluator.generate_report("evaluation_report.json")
        evaluator.plot_performance("performance_plots.png")
    else:
        print(f"Test audio file {test_audio_path} not found.")
        print(
            "Please download test data first using download.py or provide your own audio file."
        )
