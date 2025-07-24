from pathlib import Path
from typing import List, Generator, Optional
from loguru import logger

from .config import TranscriptionConfig

class AudioFileProcessor:
    """Handles discovery and processing of audio files."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
    
    def discover_audio_files(self) -> List[Path]:
        """Discover audio files based on configuration."""
        if self.config.audio_file:
            return [self.config.audio_file]
        
        if self.config.audio_folder:
            return self._find_audio_files(self.config.audio_folder)
        
        raise ValueError("No audio source specified")
    
    def _find_audio_files(self, folder: Path) -> List[Path]:
        """Find audio files in the specified folder."""
        audio_files = []
        
        pattern = "**/*" if self.config.recursive else "*"
        
        for path in folder.glob(pattern):
            if path.is_file() and self._is_audio_file(path):
                audio_files.append(path)
        
        logger.info(f"Found {len(audio_files)} audio files in {folder}")
        return sorted(audio_files)
    
    def _is_audio_file(self, path: Path) -> bool:
        """Check if a file is an audio file based on extension."""
        return path.suffix.lower() in self.config.supported_extensions
    
    def get_output_path(self, audio_path: Path, output_format: str) -> Path:
        """Generate output path for transcription file."""
        if self.config.transcription_file and self.config.audio_file:
            # Single file transcription with specified output
            return self.config.transcription_file
        
        # Generate output filename
        base_name = audio_path.stem
        extension = self._get_extension_for_format(output_format)
        output_filename = f"{base_name}_transcription.{extension}"
        
        if self.config.transcribe_folder:
            # Save to specified output folder
            return self.config.transcribe_folder / output_filename
        else:
            # Save alongside the audio file
            return audio_path.parent / output_filename
    
    def _get_extension_for_format(self, output_format: str) -> str:
        """Get file extension for output format."""
        format_extensions = {
            'md': 'md',
            'txt': 'txt',
            'json': 'json',
            'srt': 'srt',
            'vtt': 'vtt'
        }
        return format_extensions.get(output_format, 'txt')
    
    def batch_files(self, files: List[Path]) -> Generator[List[Path], None, None]:
        """Yield batches of files for processing."""
        batch_size = self.config.batch_size
        
        for i in range(0, len(files), batch_size):
            yield files[i:i + batch_size]
    
    def validate_output_path(self, output_path: Path) -> None:
        """Validate that output path is writable."""
        try:
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = output_path.parent / ".write_test"
            test_file.touch()
            test_file.unlink()
            
        except PermissionError:
            raise PermissionError(f"No write permission for: {output_path.parent}")
        except Exception as e:
            raise RuntimeError(f"Cannot write to output path {output_path}: {e}")
    
    def get_file_stats(self, files: List[Path]) -> dict:
        """Get statistics about the files to be processed."""
        total_size = sum(f.stat().st_size for f in files if f.exists())
        
        extensions = {}
        for f in files:
            ext = f.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        return {
            'total_files': len(files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_types': extensions,
            'batch_count': len(list(self.batch_files(files)))
        }