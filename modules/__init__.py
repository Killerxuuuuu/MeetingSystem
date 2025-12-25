# modules/__init__.py

"""
This package contains the core functional modules for the meeting system.
It exposes the manual diarization engine and the ASR wrapper.
"""

# Import the core classes so they can be accessed directly from the package
# This allows: "from modules import ManualDiarizer"
# Instead of: "from modules.diarizer import ManualDiarizer"
from .asr import ASRHandler
from .diarizer import ManualDiarizer
