import torch
import whisper
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .config import TranscriptionConfig

class WhisperTranscriber:
    """Core transcription class using OpenAI Whisper."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            device = self._get_device()
            logger.info(f"Loading Whisper model '{self.config.model}' on {device}")
            
            # Load model with appropriate precision
            if device == "cpu":
                # Use FP32 for CPU to avoid warning
                self.model = whisper.load_model(self.config.model, device=device)
                # Ensure model uses FP32
                if hasattr(self.model, 'half'):
                    self.model = self.model.float()
            else:
                # GPU/MPS can use FP16
                self.model = whisper.load_model(self.config.model, device=device)
                
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif self.config.device == "cpu":
            return "cpu"
        else:
            # Auto-detect best device if not explicitly set
            if torch.backends.mps.is_available():
                logger.info("Apple Silicon detected, using MPS acceleration")
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
    
    def transcribe_file(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe a single audio file."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not self._is_supported_format(audio_path):
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
        
        try:
            logger.info(f"Transcribing: {audio_path}")
            
            # Prepare transcription options
            options = self._get_transcription_options()
            
            # Perform transcription
            result = self.model.transcribe(
                str(audio_path),
                **options
            )
            
            # Add metadata
            result['metadata'] = {
                'file_path': str(audio_path),
                'file_size': audio_path.stat().st_size,
                'model': self.config.model,
                'language': result.get('language', self.config.language),
                'duration': result.get('segments', [{}])[-1].get('end', 0) if result.get('segments') else 0
            }
            
            logger.success(f"Successfully transcribed: {audio_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_path}: {e}")
            raise
    
    def _get_transcription_options(self) -> Dict[str, Any]:
        """Get transcription options for Whisper."""
        options = {}
        
        if self.config.language:
            options['language'] = self.config.language
        
        if self.config.temperature != 0.0:
            options['temperature'] = self.config.temperature
        
        # Force FP32 on CPU to avoid warning
        device = self._get_device()
        if device == "cpu":
            options['fp16'] = False
        
        # Add other sampling parameters if they differ from defaults
        decode_options = {}
        if self.config.top_p != 1.0:
            decode_options['top_p'] = self.config.top_p
        if self.config.top_k is not None:
            decode_options['top_k'] = self.config.top_k
        
        if decode_options:
            options['decode_options'] = decode_options
        
        return options
    
    def _is_supported_format(self, path: Path) -> bool:
        """Check if the file format is supported."""
        return path.suffix.lower() in self.config.supported_extensions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {}
        
        return {
            'model_name': self.config.model,
            'device': str(next(self.model.parameters()).device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'is_multilingual': self.model.is_multilingual
        }