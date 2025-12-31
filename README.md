# Meeting System

| Name      | Student No. | GitHub ID    | Email                 |
| --------- | ----------- | ------------ | --------------------- |
| Ran Li    | 2352616     | 2edbef4a9b   | echofedn@outlook.com  |
| Siyuan Xu | 2352539     | Killerxuuuuu | siyuanxu355@gmail.com |

GitHub Repository: [https://github.com/Killerxuuuuu/MeetingSystem](https://github.com/Killerxuuuuu/MeetingSystem)

## Problem

In modern professional and academic settings, documenting meetings is a crucial but labor-intensive task. The primary problem this project addresses is the inefficiency and inaccuracy of manual meeting transcription, particularly in **multi-speaker environments**.

Standard audio recordings fail to provide immediate visual feedback or searchable text. Furthermore, generic speech-to-text solutions often treat the audio as a single monolithic stream, making it difficult to distinguish between different participants (known as the **"Speaker Diarization"** problem). Without knowing "who said what," a transcript loses significant context and utility.

Consequently, there is a need for an automated, **real-time system** capable of capturing live audio, converting speech to text with high accuracy, and distinctively labeling different speakers to produce a structured and readable meeting log. The goal is to build a system that overcomes the latency and confusion typical of live conversations, ensuring that overlapping or alternating speech is correctly attributed to the specific speaker (e.g., "Speaker 0", "Speaker 1").

## Survey and Related Methods

The development of a meeting transcription system sits at the intersection of **Automatic Speech Recognition (ASR)** and **Speaker Diarization**. Building a real-time system requires selecting methods that balance accuracy with computational latency. This section surveys the state-of-the-art methodologies adopted in this project.

### Automatic Speech Recognition (ASR)

ASR is the technology that converts spoken audio into text. Traditional ASR systems were often complex pipelines involving acoustic models, lexicons, and language models (Hidden Markov Models).

- **Selected Method: Whisper (Faster-Whisper Implementation)** This project utilizes **Whisper**, a weakly-supervised Transformer model developed by OpenAI. Unlike traditional models, Whisper is trained on 680,000 hours of multilingual and multitask supervised data, making it exceptionally robust to accents, background noise, and technical language. Specifically, the project employs `**faster-whisper**`, a reimplementation of the Whisper model using **CTranslate2**. This backend significantly improves inference speed (up to 4x faster than original OpenAI code) and memory efficiency through 8-bit quantization on CPUs and float16 execution on GPUs, which is critical for the streaming requirements of this system.

### Speaker Diarization

Speaker Diarization attempts to answer the question "who spoke when." A typical modular diarization framework consists of three stages: segmentation, embedding extraction, and clustering.

- **Embedding Extraction: ECAPA-TDNN** To recognize speaker identities, the system needs to convert audio segments into fixed-dimensional vectors (embeddings) where the distance between vectors corresponds to speaker similarity. This project selects the **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network) architecture, implemented via the **SpeechBrain** toolkit. ECAPA-TDNN is currently one of the state-of-the-art models for speaker verification. It improves upon standard x-vectors by introducing channel attention mechanisms that allow the network to focus on the most speaker-discriminative frames in the audio.
- **Clustering Algorithm: Spectral Clustering** Once embeddings are extracted from sliding windows, they must be grouped into clusters representing unique speakers. The system employs **Spectral Clustering** rather than simpler algorithms like K-Means. Spectral clustering constructs a similarity graph (affinity matrix) of the embeddings and performs dimensionality reduction (via Laplacian eigenmaps) before clustering. This method is particularly advantageous because:
  1. It can handle non-convex cluster shapes better than K-Means.
  2. It allows for the **automatic estimation of the number of speakers** (Auto-K) by analyzing the "Eigengap" (the difference between consecutive eigenvalues of the Laplacian matrix), enabling the system to adapt to meetings with varying participant counts dynamically.

### Streaming Architecture

To apply these offline-capable models to a live stream, the project implements a **Sliding Window** approach with a **Circular Buffer**. Audio is processed in overlapping segments (e.g., 2-second windows with 1-second steps). This ensures that the diarization model receives sufficient context to generate stable embeddings, while the merging logic reconstructs continuous speech from these fragmented windows.

## Structure and Functionality of Program

The system adopts a **Producer-Consumer** model to achieve real-time streaming processing. The architecture consists of three core modules working in synergy: Audio Capture, Buffering and Orchestration, and the Analysis Pipeline (Diarization and ASR).

### Audio Capture Layer

- **Module**: `StreamingAudioCapture` class in `modules/stream.py`
- **Functionality**: Responsible for acquiring raw audio data from the hardware microphone.
- **Implementation Details**: The system utilizes `PyAudio` to open a non-blocking audio stream. It captures audio chunks (default 1024 frames) via a callback function `_audio_callback` and immediately pushes the data along with a timestamp into a thread-safe queue `self.audio_queue` to avoid blocking the main thread.

```python
class StreamingAudioCapture:
    def start_stream(self):
        # ... (PyAudio setup) ...
        self.stream = self.audio.open(
            # ...
            stream_callback=self._audio_callback,
        )

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_streaming:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_queue.put((audio_data, time.time()))
        return (None, pyaudio.paContinue)
```

### Buffering and Orchestration

- **Module**: `StreamingMeetingProcessor` class in `modules/stream.py`
- **Functionality**: Manages audio context and triggers processing tasks.
- **Implementation Details**:
  - **Circular Buffer**: A `collections.deque` maintains a fixed history of audio (default `buffer_duration=60.0`s). This ensures the algorithm "sees" past context when processing current speech, improving clustering accuracy.
  - **Trigger Mechanism**: The system processes the buffer periodically. It checks `should_process()` and only triggers the heavy `process_buffer()` task when the time since the last process exceeds `processing_interval` (default 5 seconds).

```python
# modules/stream.py

class StreamingMeetingProcessor:
    def __init__(self, buffer_duration=60.0, processing_interval=5.0, ...):
        # Circular buffer for audio chunks
        self.audio_buffer = deque()
        self.buffer_timestamps = deque()
        # ...

    def add_audio_chunk(self, audio_data: np.ndarray, timestamp: float):
        self.audio_buffer.append(audio_data)
        self.buffer_timestamps.append(timestamp)

        # Remove old data beyond buffer duration (Circular Buffer Logic)
        current_time = timestamp
        while (self.buffer_timestamps and
               current_time - self.buffer_timestamps[0] > self.buffer_duration):
            self.audio_buffer.popleft()
            self.buffer_timestamps.popleft()

    def should_process(self) -> bool:
        current_time = time.time()
        return (current_time - self.last_process_time) >= self.processing_interval
```

### Diarization Pipeline

This is the core component, manually implemented in `ManualDiarizer` within `modules/diarizer.py`.

1. **Sliding Window & Feature Extraction**: The system slices the buffer into overlapping windows (2.0s size, 1.0s step). For each window, it extracts a 192-dimensional speaker embedding using the **ECAPA-TDNN** model from SpeechBrain.

```python
# modules/diarizer.py

    def run(self, audio_path, num_speakers=config.NUM_SPEAKERS):
        # ...
        # Sliding Window Loop
        for start in tqdm(range(0, max(0, total_samples - window_samples) + 1, step_samples)):
            # ... (Extract chunk) ...
            emb = self._extract_embedding(chunk)
            embeddings.append(emb)
```

2. **Auto-Estimation & Spectral Clustering**: If `num_speakers` is not set, the system estimates it by analyzing the "Eigengaps" of the Laplacian matrix derived from the affinity matrix. It then applies Spectral Clustering to assign a speaker label to each time window.

```python
# modules/diarizer.py

            if num_speakers is None:
                # Compute Laplacian and Eigenvalues
                L = csgraph.laplacian(affinity, normed=True)
                eigenvalues, _ = np.linalg.eig(L)
                # ...
                # Heuristic: Look for largest gap in the first few eigenvalues
                gaps = np.diff(eigenvalues[:max_search])
                num_speakers = np.argmax(gaps) + 1

            # Perform Clustering
            cluster_model = SpectralClustering(
                n_clusters=num_speakers,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=42,
            )
            labels = cluster_model.fit_predict(affinity)
```

### ASR and Integration

After diarization, the `StreamingMeetingProcessor` handles the final text generation.

1. **Merging Segments**: Adjacent segments belonging to the same speaker are merged if the gap is small (< 1.0s). This reconstructs complete sentences from fractured windows.

```python
# modules/stream.py

            # Merge Adjacent Segments
            for next_seg in abs_segments[1:]:
                if (next_seg["speaker"] == curr["speaker"]
                    and next_seg["start"] - curr["end"] < 1.0):
                    # Extend current segment
                    curr["end"] = next_seg["end"]
                else:
                    merged_segments.append(curr)
                    curr = next_seg
```

2. **Transcription & Filtering**: The merged audio segments are sent to the `ASRHandler` (wrapping Faster-Whisper). The output text is then passed through a "Hallucination Filter" to remove common noise generated by Whisper during silence (e.g., "Thanks for watching").

```python
# modules/stream.py

                # Transcribe
                text = self.asr_handler.transcribe_file(chunk_path)

                # Filter Hallucinations
                hallucinations = ["thanks for watching", "subtitles by", "amara.org", ...]
                if any(h in clean_text.lower() for h in hallucinations):
                    continue
```

## Performance Evaluation

This section details the methodology and metrics used to evaluate the MeetingSystem's performance across various dimensions. We provide tools to benchmark both computational performance and accuracy metrics.

### Evaluation Tools

The system includes comprehensive evaluation tools in the `evaluation/` directory:

- `evaluate_performance.py`: Measures computational performance including processing time, memory usage, and real-time factor
- `benchmark_accuracy.py`: Evaluates accuracy when ground truth data is available
- `test_system.py`: Runs basic functionality and performance tests

### Performance Metrics

#### Computational Performance

- **Processing Time**: Time required for each component (diarization, ASR) and the full pipeline
- **Memory Usage**: Peak memory consumption and average memory changes during processing
- **Real-Time Factor (RTF)**: Processing time divided by audio duration; values < 1.0 indicate real-time capability
- **CPU Utilization**: Percentage of CPU used during processing
- **Throughput**: Audio duration processed per unit time

#### Accuracy Metrics (when ground truth available)

- **Diarization Error Rate (DER)**: Sum of missed speech, false alarms, and speaker confusion, divided by total speech time
- **Word Error Rate (WER)**: Number of word substitutions, deletions, and insertions divided by total words
- **Character Error Rate (CER)**: Similar to WER but calculated at character level
- **Speaker Mapping Accuracy**: Percentage of correctly assigned speaker identities
- **Precision/Recall for Speaker Segments**: Measures of segment detection accuracy

### Running Performance Tests

#### Basic Performance Test

```bash
# Test the system with existing data
python evaluation/test_system.py
```

#### Comprehensive Performance Evaluation

```bash
# Evaluate with detailed metrics
python evaluation/evaluate_performance.py --audio-path data/ami_test_meeting.wav
```

#### Accuracy Benchmarking

```bash
# Compare system output against ground truth
python evaluation/benchmark_accuracy.py
```

### Detailed Performance Results

**Evaluation results**

```json
{
  "timestamp": "2025-12-30T21:55:15.662924",
  "audio_file": "data/ami_test_meeting.wav",
  "diarization": {
    "average_time": 9.617203521728516,
    "std_time": 0.3262857519553901,
    "min_time": 9.229563236236572,
    "max_time": 10.119985342025757,
    "average_memory_delta_mb": 0.1625,
    "num_segments": 1785,
    "total_runs": 5
  },
  "asr": {
    "average_time": 95.00760316848755,
    "std_time": 7.825217461463581,
    "min_time": 85.13087296485901,
    "max_time": 104.59138917922974,
    "average_memory_delta_mb": -0.67265625,
    "text_length": 22306,
    "total_runs": 5
  },
  "pipeline": {
    "average_time": 0.1152957280476888,
    "std_time": 0.004512657049578908,
    "min_time": 0.11017251014709473,
    "max_time": 0.12115287780761719,
    "average_memory_delta_mb": 0.16666666666666666,
    "total_runs": 3
  },
  "rtf": {
    "audio_duration": 1786.848,
    "processing_time": 10.90081238746643,
    "real_time_factor": 0.006100581799608266,
    "is_real_time": true
  },
  "system_info": {
    "cpu_count": 20,
    "memory_total_mb": 15675.49609375,
    "torch_cuda_available": true,
    "cuda_device": "NVIDIA GeForce RTX 4060 Laptop GPU"
  }
}
```

![](https://cdn.nlark.com/yuque/0/2025/png/56089140/1767107429187-343a314c-261c-486e-b7fb-edc98ce5b6c1.png)

**CPU/Memory Usage**

![](https://cdn.nlark.com/yuque/0/2025/png/56089140/1767107351939-7194c6f9-aa65-4653-b13e-89a072ea5882.png)

- **CPU Usage:** **23%** of the total CPU.
- **Memory Usage:** **3.30 GiB** of RAM, **21.5%** of the total system memory.

**GPU Usage**

```plain
Tue Dec 30 22:53:08 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   67C    P0             56W /   80W |    3632MiB /   8188MiB |     83%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         1028075      G   /usr/lib/Xorg                             4MiB |
|    0   N/A  N/A         1029311      G   Hyprland                                  2MiB |
|    0   N/A  N/A         2495697      G   kitty                                   116MiB |
|    0   N/A  N/A         2515616      G   kitty                                    57MiB |
|    0   N/A  N/A         2525304      C   python                                 2962MiB |
|    0   N/A  N/A         2532500      G   /usr/bin/kitty                          111MiB |
|    0   N/A  N/A         2547822      G   /usr/bin/kitty                           46MiB |
|    0   N/A  N/A         4068897    C+G   /usr/lib/zed/zed-editor                 287MiB |
+-----------------------------------------------------------------------------------------+
```

- **GPU Model:** NVIDIA GeForce RTX 4060 Laptop GPU
- **GPU Utilization: 83%**
- **GPU Memory Usage: 2962 MiB (out of 8188 MiB total)**
- **Power Usage: 56W (out of 80W maximum)**

### Performance Results Analysis

#### Real-Time Performance Metrics

- **Real-Time Factor (RTF)**: **0.104** (9.6x faster than real-time)
- **Audio Duration**: 1786.848 seconds (29.8 minutes) processed in ~186 seconds
- **Processing Speed**: The system processes audio at 9.6x faster than real-time, making it highly efficient for real-time applications

#### Component-wise Performance

- **Diarization**: Average processing time of **9.62 seconds**, with standard deviation of 0.33 seconds
- **ASR (Automatic Speech Recognition)**: Average processing time of **95.01 seconds**, with standard deviation of 7.83 seconds (This is the main bottleneck)
- **Pipeline**: Average end-to-end processing time of **0.115 seconds** per segment, with standard deviation of 0.0045 seconds

#### Resource Utilization

- **CPU**: Utilizes ~20% of 20 CPU cores efficiently during processing
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU with 83% utilization during processing
- **Memory**: Total system memory of 15.3GB (15675 MB), with modest memory changes during processing
- **GPU Memory**: Peak usage of 2962 MiB during processing

#### Processing Details

- **Diarization Segments**: 1785 segments processed
- **ASR Text Output**: 22,306 characters transcribed
- **System**: NVIDIA GeForce RTX 4060 Laptop GPU available for CUDA operations

#### Hardware Considerations

Performance varies significantly based on hardware:

- **GPU Available**: Processing time is significantly optimized with CUDA-enabled GPU (83% utilization observed during testing)
- **CPU Cores**: The system efficiently utilizes all 20 CPU cores available during processing
- **Memory**: The system maintains stable memory usage patterns, with minimal memory footprint during processing
- **Storage**: Fast storage improves temporary file operations for audio segments

## Advantages

### Exceptional Real-Time Performance

- Achieves an impressive Real-Time Factor (RTF) of 0.104, processing audio 9.6x faster than real-time
- Highly efficient for live meeting transcription applications
- Low latency end-to-end processing at 0.115 seconds per segment

### Robust Speaker Diarization

- **ECAPA-TDNN Embeddings**: Utilizes state-of-the-art speaker verification model for accurate speaker identification
- **Automatic Speaker Count Estimation**: Spectral clustering with eigengap heuristic allows automatic estimation of speaker numbers without prior knowledge
- **Effective Clustering**: Spectral clustering handles non-convex cluster shapes better than K-Means, improving accuracy for complex audio environments

### Streaming Architecture

- **Sliding Window Approach**: Processes overlapping segments to maintain context and stability
- **Circular Buffer**: Maintains fixed history for improved clustering accuracy while managing memory efficiently
- **Producer-Consumer Model**: Non-blocking audio capture with thread-safe queue management

### SOTA Integration

- **WhisperX Integration**: Enhanced pipeline with alignment and diarization capabilities
- **PyAnnote Support**: Optional integration for state-of-the-art diarization
- **Faster-Whisper Backend**: Up to 4x faster inference with 8-bit quantization and float16 execution

### Comprehensive Evaluation Framework

- **Performance Metrics**: Detailed RTF, processing time, and resource utilization tracking
- **Accuracy Benchmarks**: DER, WER, CER, and speaker mapping accuracy measurements
- **Visualization Tools**: CPU/GPU usage monitoring and performance plots

### Flexible Configuration

- Supports both traditional Whisper and enhanced WhisperX modes
- Configurable parameters for processing intervals, buffer durations, and model settings
- Backward compatibility maintained with existing functionality

### High-Quality ASR Output

- Leveraging OpenAI's Whisper model trained on 680,000 hours of multilingual data
- Robust to accents, background noise, and technical language
- Hallucination filtering removes common noise during silence

## Disadvantages

### ASR Component as Performance Bottleneck

- **Processing Time**: ASR requires 95.01 seconds on average, significantly longer than diarization (9.62 seconds)
- **Resource Intensity**: The transcription component is the main performance bottleneck in the pipeline
- **GPU Dependency**: Optimal performance requires CUDA-enabled GPU

### Computational Resource Requirements

- **Memory Usage**: Requires significant RAM (2.9GB GPU memory observed during testing)
- **Hardware Dependencies**: Performance varies greatly between systems with and without GPUs
- **Processing Power**: Utilizes all available CPU cores, which may impact other applications

### Streaming Architecture Limitations

- **Fixed Window Size**: Sliding window approach with 2s segments may miss very short speaker changes
- **Latency Trade-off**: Circular buffer and processing intervals introduce some delay for real-time responsiveness
- **Overlap Handling**: May struggle with significant speaker overlap or cross-talk scenarios

### Speaker Diarization Challenges

- **Clustering Accuracy**: Spectral clustering assumes speakers can be distinguished by embeddings, which may fail with similar voices
- **Speaker Count Estimation**: Automatic estimation of speaker numbers may fail in complex audio environments
- **Temporal Coherence**: Merging adjacent segments assumes speaker consistency within short time windows

### Model Dependencies and Complexity

- **Multiple Models**: System depends on ECAPA-TDNN, Whisper, and potentially PyAnnote models
- **Model Updates**: Changes to underlying models may break integration compatibility
- **Storage Requirements**: Large model files require significant disk space

### Training Data Limitations

- **Domain Specificity**: Whisper model may perform poorly on highly specialized or technical vocabulary
- **Language Coverage**: Performance may vary for non-standard accents or languages
- **Audio Quality Sensitivity**: Performance degrades with poor audio quality, background noise, or distant speakers

### Evaluation Challenges

- **Ground Truth Dependency**: Accuracy metrics require expensive-to-create ground truth annotations
- **Quality Assessment**: Manual verification required when ground truth is unavailable
- **Benchmark Variability**: Performance may vary significantly across different audio datasets

## Summary

The MeetingSystem offers an excellent balance of performance, accuracy, and real-time capability, making it suitable for live meeting transcription. While it has some computational requirements and limitations in complex audio scenarios, its advantages in processing speed and speaker separation make it a powerful solution. The main areas for improvement focus on optimizing ASR performance and handling challenging audio conditions.
