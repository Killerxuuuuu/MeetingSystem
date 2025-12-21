from faster_whisper import WhisperModel
import config

class ASRHandler:
    def __init__(self):
        print(f"[Init] Loading Whisper ASR Model ({config.ASR_MODEL_SIZE})...")
        # Load the model with optimization (float16 for GPU, int8 for CPU)
        self.model = WhisperModel(
            config.ASR_MODEL_SIZE, 
            device=config.DEVICE, 
            compute_type="float16" if config.DEVICE=="cuda" else "int8"
        )
    
    def transcribe_file(self, file_path):
        """
        Transcribe a specific audio file (segment) into text.
        """
        # beam_size=5 provides better accuracy at the cost of speed
        segments, _ = self.model.transcribe(file_path, beam_size=5, language="en") # Change "zh" to "en" for English
        
        # Combine all segments into a single string
        return "".join([s.text for s in segments])