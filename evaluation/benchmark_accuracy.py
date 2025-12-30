"""
Accuracy Benchmarking Script for MeetingSystem

This script evaluates the accuracy of the MeetingSystem when ground truth data is available.
It calculates metrics like Diarization Error Rate (DER), Word Error Rate (WER), etc.
"""

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from jiwer import cer, wer  # These would need to be installed: pip install jiwer


@dataclass
class GroundTruthSegment:
    start_time: float
    end_time: float
    speaker: str
    text: str


@dataclass
class HypothesisSegment:
    start_time: float
    end_time: float
    speaker: str
    text: str


class AccuracyBenchmark:
    def __init__(self):
        self.metrics = {}

    def load_ground_truth(self, gt_path: str) -> list[GroundTruthSegment]:
        """Load ground truth data from JSON or CSV"""
        gt_segments = []

        if gt_path.endswith(".json"):
            with open(gt_path, encoding="utf-8") as f:
                gt_data = json.load(f)
                for item in gt_data:
                    gt_segments.append(
                        GroundTruthSegment(
                            start_time=item["start"],
                            end_time=item["end"],
                            speaker=item["speaker"],
                            text=item["text"],
                        )
                    )
        elif gt_path.endswith(".rttm"):
            # Parse RTTM format commonly used for diarization evaluation
            gt_segments = self._parse_rttm(gt_path)
        elif gt_path.endswith(".csv"):
            # Parse CSV format
            gt_segments = self._parse_csv(gt_path)

        return gt_segments

    def _parse_rttm(self, rttm_path: str) -> list[GroundTruthSegment]:
        """Parse RTTM (Rich Transcription Time Marked) format"""
        segments = []
        with open(rttm_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if (
                    len(parts) >= 6
                ):  # RTTM format: TYPE FILE_ID CHAN START DUR WORD [CONF] [SLAT] SPKR_NAME [CONF]
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    segments.append(
                        GroundTruthSegment(
                            start_time=start,
                            end_time=start + duration,
                            speaker=speaker,
                            text="",  # RTTM typically doesn't include text
                        )
                    )
        return segments

    def _parse_csv(self, csv_path: str) -> list[GroundTruthSegment]:
        """Parse CSV format"""
        import csv

        segments = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                segments.append(
                    GroundTruthSegment(
                        start_time=float(row["start"]),
                        end_time=float(row["end"]),
                        speaker=row["speaker"],
                        text=row.get("text", ""),
                    )
                )
        return segments

    def calculate_diarization_error_rate(
        self,
        ground_truth: list[GroundTruthSegment],
        hypothesis: list[HypothesisSegment],
    ) -> dict:
        """Calculate Diarization Error Rate (DER)"""
        # Convert segments to time-sparse representation
        # This is a simplified version - full DER calculation is complex
        # and typically requires specialized libraries like md-eval or pyannote.metrics

        # For now, implement a basic overlap-based approach
        total_time = max(
            max([seg.end_time for seg in ground_truth], default=0),
            max([seg.end_time for seg in hypothesis], default=0),
        )

        # Create time grids
        gt_grid = np.zeros(int(total_time * 100))  # 10ms resolution
        hyp_grid = np.zeros(int(total_time * 100))

        for seg in ground_truth:
            start_idx = int(seg.start_time * 100)
            end_idx = int(seg.end_time * 100)
            # For simplicity, we'll just mark speaker presence without IDs
            if start_idx < len(gt_grid):
                gt_grid[start_idx : min(end_idx, len(gt_grid))] = 1

        for seg in hypothesis:
            start_idx = int(seg.start_time * 100)
            end_idx = int(seg.end_time * 100)
            if start_idx < len(hyp_grid):
                hyp_grid[start_idx : min(end_idx, len(hyp_grid))] = 1

        # Calculate DER components
        total_frames = len(gt_grid)
        missed_detection = np.sum((gt_grid == 1) & (hyp_grid == 0))
        false_alarm = np.sum((gt_grid == 0) & (hyp_grid == 1))
        speaker_confusion = np.sum(
            (gt_grid == 1) & (hyp_grid == 1) & (gt_grid != hyp_grid)
        )  # Simplified

        der = (
            (missed_detection + false_alarm + speaker_confusion) / total_frames
            if total_frames > 0
            else 0
        )

        der_metrics = {
            "diarization_error_rate": der,
            "missed_detection_rate": missed_detection / total_frames
            if total_frames > 0
            else 0,
            "false_alarm_rate": false_alarm / total_frames if total_frames > 0 else 0,
            "speaker_confusion_rate": speaker_confusion / total_frames
            if total_frames > 0
            else 0,
            "total_frames": total_frames,
            "missed_frames": missed_detection,
            "false_alarm_frames": false_alarm,
            "confusion_frames": speaker_confusion,
        }

        return der_metrics

    def calculate_word_error_rate(
        self,
        ground_truth: list[GroundTruthSegment],
        hypothesis: list[HypothesisSegment],
    ) -> dict:
        """Calculate Word Error Rate (WER) and Character Error Rate (CER)"""
        # Combine all ground truth text
        gt_text = " ".join([seg.text for seg in ground_truth if seg.text.strip()])
        hyp_text = " ".join([seg.text for seg in hypothesis if seg.text.strip()])

        # Calculate WER and CER
        word_error_rate = wer(gt_text, hyp_text) if gt_text and hyp_text else 0
        char_error_rate = cer(gt_text, hyp_text) if gt_text and hyp_text else 0

        # Calculate additional metrics
        gt_words = len(gt_text.split()) if gt_text else 0
        hyp_words = len(hyp_text.split()) if hyp_text else 0

        wer_metrics = {
            "word_error_rate": word_error_rate,
            "character_error_rate": char_error_rate,
            "ground_truth_word_count": gt_words,
            "hypothesis_word_count": hyp_words,
            "gt_text_length": len(gt_text),
            "hyp_text_length": len(hyp_text),
        }

        return wer_metrics

    def calculate_speaker_mapping(
        self,
        ground_truth: list[GroundTruthSegment],
        hypothesis: list[HypothesisSegment],
    ) -> dict:
        """Calculate optimal speaker mapping and agreement"""
        # Create speaker co-occurrence matrix
        gt_speakers = {seg.speaker for seg in ground_truth}
        hyp_speakers = {seg.speaker for seg in hypothesis}

        # Simple overlap-based mapping (this is a simplified approach)
        speaker_mapping = {}
        total_overlap = defaultdict(lambda: defaultdict(float))

        # Calculate overlaps between each GT and Hyp speaker
        for gt_seg in ground_truth:
            for hyp_seg in hypothesis:
                # Calculate time overlap
                overlap_start = max(gt_seg.start_time, hyp_seg.start_time)
                overlap_end = min(gt_seg.end_time, hyp_seg.end_time)

                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    total_overlap[gt_seg.speaker][hyp_seg.speaker] += overlap_duration

        # Create best mapping (greedy approach)
        used_hyp_speakers = set()
        for gt_speaker in gt_speakers:
            if gt_speaker in total_overlap:
                best_hyp = max(
                    total_overlap[gt_speaker].items(),
                    key=lambda x: x[1],
                    default=(None, 0),
                )
                if best_hyp[0] not in used_hyp_speakers:
                    speaker_mapping[gt_speaker] = best_hyp[0]
                    used_hyp_speakers.add(best_hyp[0])

        # Calculate speaker agreement
        total_time = sum(seg.end_time - seg.start_time for seg in ground_truth)
        correctly_mapped_time = 0

        for gt_seg in ground_truth:
            if gt_seg.speaker in speaker_mapping:
                expected_hyp_speaker = speaker_mapping[gt_seg.speaker]
                for hyp_seg in hypothesis:
                    overlap_start = max(gt_seg.start_time, hyp_seg.start_time)
                    overlap_end = min(gt_seg.end_time, hyp_seg.end_time)

                    if (
                        overlap_start < overlap_end
                        and hyp_seg.speaker == expected_hyp_speaker
                    ):
                        correctly_mapped_time += overlap_end - overlap_start

        speaker_accuracy = correctly_mapped_time / total_time if total_time > 0 else 0

        mapping_metrics = {
            "speaker_mapping": dict(speaker_mapping),
            "speaker_accuracy": speaker_accuracy,
            "ground_truth_speakers": list(gt_speakers),
            "hypothesis_speakers": list(hyp_speakers),
            "correctly_mapped_time": correctly_mapped_time,
            "total_time": total_time,
        }

        return mapping_metrics

    def run_accuracy_benchmark(
        self,
        audio_path: str,
        ground_truth_path: str,
        output_path: str = "accuracy_report.json",
    ) -> dict:
        """Run the complete accuracy benchmark"""
        print("Running accuracy benchmark...")
        print(f"Audio: {audio_path}")
        print(f"Ground Truth: {ground_truth_path}")

        # Load ground truth
        ground_truth = self.load_ground_truth(ground_truth_path)
        print(f"Loaded {len(ground_truth)} ground truth segments")

        # Process audio with MeetingSystem to get hypothesis
        # For this example, we'll simulate with a manual process
        # In practice, you'd run the actual system
        print("Processing audio to generate hypothesis...")

        # This is where you'd integrate the actual MeetingSystem
        # For demonstration, creating mock hypothesis
        hypothesis = self._generate_mock_hypothesis(ground_truth)

        # Calculate metrics
        der_metrics = self.calculate_diarization_error_rate(ground_truth, hypothesis)
        wer_metrics = self.calculate_word_error_rate(ground_truth, hypothesis)
        mapping_metrics = self.calculate_speaker_mapping(ground_truth, hypothesis)

        # Combine all metrics
        results = {
            "audio_file": audio_path,
            "ground_truth_file": ground_truth_path,
            "diarization": der_metrics,
            "asr_quality": wer_metrics,
            "speaker_mapping": mapping_metrics,
            "summary": {
                "aggregate_score": (
                    (1 - der_metrics["diarization_error_rate"]) * 0.5
                    + (1 - wer_metrics["word_error_rate"]) * 0.5
                ),
                "diarization_quality": 1 - der_metrics["diarization_error_rate"],
                "asr_quality": 1 - wer_metrics["word_error_rate"],
            },
        }

        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Accuracy benchmark complete. Results saved to {output_path}")

        # Print summary
        self._print_accuracy_summary(results)

        return results

    def _generate_mock_hypothesis(
        self, ground_truth: list[GroundTruthSegment]
    ) -> list[HypothesisSegment]:
        """Generate mock hypothesis for demonstration purposes"""
        # In a real implementation, this would run the actual MeetingSystem
        # For now, create slightly perturbed versions of ground truth
        hypothesis = []
        speaker_offset = 0  # Simulate potential speaker ID mismatches

        for gt_seg in ground_truth:
            # Create hypothesis segment with potential errors
            hyp_seg = HypothesisSegment(
                start_time=gt_seg.start_time
                + np.random.uniform(-0.1, 0.1),  # Timing error
                end_time=gt_seg.end_time + np.random.uniform(-0.1, 0.1),  # Timing error
                speaker=f"Speaker_{(int(gt_seg.speaker.split('_')[-1]) + speaker_offset) % 3 if '_' in gt_seg.speaker else int(gt_seg.speaker.split(' ')[-1]) + speaker_offset}",
                text=self._perturb_text(gt_seg.text),  # Text errors
            )

            # Ensure valid times
            hyp_seg.start_time = max(0, hyp_seg.start_time)
            hyp_seg.end_time = max(hyp_seg.start_time + 0.1, hyp_seg.end_time)

            hypothesis.append(hyp_seg)

        return hypothesis

    def _perturb_text(self, text: str) -> str:
        """Simulate ASR errors in text"""
        if not text:
            return text

        # Simple perturbation: occasionally remove words or add errors
        import random

        words = text.split()
        if len(words) > 3 and random.random() < 0.1:  # 10% chance of word error
            idx = random.randint(0, len(words) - 1)
            # Remove word or add substitution error
            if random.random() < 0.5:
                words.pop(idx)
            else:
                # Substitution error
                words[idx] = (
                    words[idx][: len(words[idx]) // 2] + "error"
                )  # Truncate and add error

        return " ".join(words)

    def _print_accuracy_summary(self, results: dict):
        """Print a summary of accuracy results"""
        print("\n" + "=" * 60)
        print("ACCURACY BENCHMARK SUMMARY")
        print("=" * 60)

        print("\nDiarization Performance:")
        print(
            f"  Diarization Error Rate (DER): {results['diarization']['diarization_error_rate']:.3f}"
        )
        print(
            f"  Missed Detection Rate: {results['diarization']['missed_detection_rate']:.3f}"
        )
        print(f"  False Alarm Rate: {results['diarization']['false_alarm_rate']:.3f}")
        print(
            f"  Speaker Confusion Rate: {results['diarization']['speaker_confusion_rate']:.3f}"
        )

        print("\nASR Quality:")
        print(
            f"  Word Error Rate (WER): {results['asr_quality']['word_error_rate']:.3f}"
        )
        print(
            f"  Character Error Rate (CER): {results['asr_quality']['character_error_rate']:.3f}"
        )
        print(
            f"  Ground Truth Words: {results['asr_quality']['ground_truth_word_count']}"
        )
        print(f"  Hypothesis Words: {results['asr_quality']['hypothesis_word_count']}")

        print("\nSpeaker Mapping:")
        print(
            f"  Speaker Accuracy: {results['speaker_mapping']['speaker_accuracy']:.3f}"
        )
        print(
            f"  Ground Truth Speakers: {results['speaker_mapping']['ground_truth_speakers']}"
        )
        print(
            f"  Hypothesis Speakers: {results['speaker_mapping']['hypothesis_speakers']}"
        )

        print("\nOverall Scores:")
        print(f"  Diarization Quality: {results['summary']['diarization_quality']:.3f}")
        print(f"  ASR Quality: {results['summary']['asr_quality']:.3f}")
        print(f"  Aggregate Score: {results['summary']['aggregate_score']:.3f}")

        print("=" * 60)


def main():
    # Example usage
    benchmark = AccuracyBenchmark()

    # Example: if you have ground truth files
    # benchmark.run_accuracy_benchmark(
    #     audio_path="data/test_audio.wav",
    #     ground_truth_path="data/test_ground_truth.json",
    #     output_path="accuracy_report.json"
    # )


if __name__ == "__main__":
    main()
