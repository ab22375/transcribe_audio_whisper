import pytest
from pathlib import Path
from pydantic import ValidationError

from src.config import TranscriptionConfig

class TestTranscriptionConfig:
    """Test cases for TranscriptionConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TranscriptionConfig(audio_file=Path("test.mp3"))
        
        assert config.model == "base"
        assert config.language == "en"
        assert config.output_format == "md"
        assert config.device == "cpu"
        assert config.temperature == 0.0
        assert config.top_p == 1.0
        assert config.top_k is None
        assert config.batch_size == 1
        assert config.max_length is None
        assert config.recursive is False
        assert config.verbose is False
    
    def test_valid_config_with_audio_file(self):
        """Test valid configuration with audio file."""
        config = TranscriptionConfig(
            model="large",
            language="fr",
            audio_file=Path("audio.wav"),
            output_format="json",
            device="cuda",
            verbose=True
        )
        
        assert config.model == "large"
        assert config.language == "fr"
        assert config.output_format == "json"
        assert config.device == "cuda"
        assert config.verbose is True
    
    def test_valid_config_with_audio_folder(self):
        """Test valid configuration with audio folder."""
        config = TranscriptionConfig(
            audio_folder=Path("/tmp"),
            recursive=True,
            batch_size=5
        )
        
        assert config.audio_folder == Path("/tmp")
        assert config.recursive is True
        assert config.batch_size == 5
    
    def test_invalid_temperature(self):
        """Test invalid temperature values."""
        with pytest.raises(ValidationError):
            TranscriptionConfig(audio_file=Path("test.mp3"), temperature=-0.1)
        
        with pytest.raises(ValidationError):
            TranscriptionConfig(audio_file=Path("test.mp3"), temperature=1.1)
    
    def test_invalid_top_p(self):
        """Test invalid top_p values."""
        with pytest.raises(ValidationError):
            TranscriptionConfig(audio_file=Path("test.mp3"), top_p=-0.1)
        
        with pytest.raises(ValidationError):
            TranscriptionConfig(audio_file=Path("test.mp3"), top_p=1.1)
    
    def test_invalid_batch_size(self):
        """Test invalid batch size."""
        with pytest.raises(ValidationError):
            TranscriptionConfig(audio_file=Path("test.mp3"), batch_size=0)
        
        with pytest.raises(ValidationError):
            TranscriptionConfig(audio_file=Path("test.mp3"), batch_size=-1)
    
    def test_invalid_max_length(self):
        """Test invalid max length."""
        with pytest.raises(ValidationError):
            TranscriptionConfig(audio_file=Path("test.mp3"), max_length=0)
        
        with pytest.raises(ValidationError):
            TranscriptionConfig(audio_file=Path("test.mp3"), max_length=-1)
    
    def test_invalid_top_k(self):
        """Test invalid top_k values."""
        with pytest.raises(ValidationError):
            TranscriptionConfig(audio_file=Path("test.mp3"), top_k=0)
        
        with pytest.raises(ValidationError):
            TranscriptionConfig(audio_file=Path("test.mp3"), top_k=-1)
    
    def test_supported_extensions(self):
        """Test supported audio extensions."""
        config = TranscriptionConfig(audio_file=Path("test.mp3"))
        extensions = config.supported_extensions
        
        expected_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.mp4', '.mkv', '.avi'}
        assert extensions == expected_extensions
    
    def test_validate_paths_no_source(self):
        """Test validation with no audio source."""
        config = TranscriptionConfig()
        
        with pytest.raises(ValueError, match="Either audio_file or audio_folder must be specified"):
            config.validate_paths()
    
    def test_validate_paths_both_sources(self):
        """Test validation with both audio sources."""
        config = TranscriptionConfig(
            audio_file=Path("test.mp3"),
            audio_folder=Path("/tmp")
        )
        
        with pytest.raises(ValueError, match="Cannot specify both audio_file and audio_folder"):
            config.validate_paths()
    
    def test_validate_paths_nonexistent_file(self):
        """Test validation with nonexistent audio file."""
        config = TranscriptionConfig(audio_file=Path("nonexistent.mp3"))
        
        with pytest.raises(ValueError, match="Audio file does not exist"):
            config.validate_paths()
    
    def test_validate_paths_nonexistent_folder(self):
        """Test validation with nonexistent audio folder."""
        config = TranscriptionConfig(audio_folder=Path("/nonexistent"))
        
        with pytest.raises(ValueError, match="Audio folder does not exist"):
            config.validate_paths()
    
    def test_validate_paths_creates_output_folder(self, tmp_path):
        """Test that output folder is created if it doesn't exist."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        output_folder = tmp_path / "output"
        
        config = TranscriptionConfig(
            audio_file=audio_file,
            transcribe_folder=output_folder
        )
        
        assert not output_folder.exists()
        config.validate_paths()
        assert output_folder.exists()