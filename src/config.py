from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

OutputFormat = Literal["md", "txt", "json", "srt", "vtt"]
Device = Literal["cpu", "cuda", "mps"]

class TranscriptionConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    # Whisper model parameters
    model: str = Field(default="base", description="Whisper model to use")
    language: Optional[str] = Field(default="en", description="Language for transcription")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling parameter")
    
    # Processing parameters
    device: Device = Field(default="cpu", description="Device for processing")
    batch_size: int = Field(default=1, ge=1, description="Batch size for processing")
    max_length: Optional[int] = Field(default=None, ge=1, description="Maximum transcription length")
    
    # File handling parameters
    audio_file: Optional[Path] = Field(default=None, description="Single audio file to transcribe")
    audio_folder: Optional[Path] = Field(default=None, description="Folder to search for audio files")
    transcribe_folder: Optional[Path] = Field(default=None, description="Output folder for transcriptions")
    transcription_file: Optional[Path] = Field(default=None, description="Output file for single transcription")
    recursive: bool = Field(default=False, description="Recursively search audio folder")
    
    # Output parameters
    output_format: OutputFormat = Field(default="md", description="Output format")
    verbose: bool = Field(default=False, description="Enable verbose output")
    
    def validate_paths(self) -> None:
        """Validate that required paths exist and configurations are consistent."""
        if self.audio_file and not self.audio_file.exists():
            raise ValueError(f"Audio file does not exist: {self.audio_file}")
        
        if self.audio_folder and not self.audio_folder.exists():
            raise ValueError(f"Audio folder does not exist: {self.audio_folder}")
        
        if not self.audio_file and not self.audio_folder:
            raise ValueError("Either audio_file or audio_folder must be specified")
        
        if self.audio_file and self.audio_folder:
            raise ValueError("Cannot specify both audio_file and audio_folder")
        
        if self.transcribe_folder and not self.transcribe_folder.exists():
            self.transcribe_folder.mkdir(parents=True, exist_ok=True)
    
    @property
    def supported_extensions(self) -> set[str]:
        """Get supported audio file extensions."""
        return {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.mp4', '.mkv', '.avi'}