import pytest
from pathlib import Path
from unittest.mock import Mock

from src.config import TranscriptionConfig
from src.file_processor import AudioFileProcessor

class TestAudioFileProcessor:
    """Test cases for AudioFileProcessor."""
    
    @pytest.fixture
    def config_with_file(self, tmp_path):
        """Create config with single audio file."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        return TranscriptionConfig(audio_file=audio_file)
    
    @pytest.fixture
    def config_with_folder(self, tmp_path):
        """Create config with audio folder."""
        return TranscriptionConfig(audio_folder=tmp_path)
    
    @pytest.fixture
    def audio_files(self, tmp_path):
        """Create test audio files."""
        files = []
        for name in ["audio1.mp3", "audio2.wav", "audio3.m4a", "not_audio.txt"]:
            file_path = tmp_path / name
            file_path.touch()
            files.append(file_path)
        return files
    
    def test_discover_audio_files_single_file(self, config_with_file):
        """Test discovering single audio file."""
        processor = AudioFileProcessor(config_with_file)
        files = processor.discover_audio_files()
        
        assert len(files) == 1
        assert files[0] == config_with_file.audio_file
    
    def test_discover_audio_files_folder(self, config_with_folder, audio_files):
        """Test discovering audio files in folder."""
        processor = AudioFileProcessor(config_with_folder)
        files = processor.discover_audio_files()
        
        # Should find 3 audio files, not the txt file
        assert len(files) == 3
        audio_extensions = {'.mp3', '.wav', '.m4a'}
        found_extensions = {f.suffix for f in files}
        assert found_extensions == audio_extensions
    
    def test_discover_audio_files_recursive(self, tmp_path):
        """Test recursive audio file discovery."""
        # Create nested structure
        subdir = tmp_path / "subfolder"
        subdir.mkdir()
        
        (tmp_path / "audio1.mp3").touch()
        (subdir / "audio2.wav").touch()
        
        config = TranscriptionConfig(audio_folder=tmp_path, recursive=True)
        processor = AudioFileProcessor(config)
        files = processor.discover_audio_files()
        
        assert len(files) == 2
        assert any(f.name == "audio1.mp3" for f in files)
        assert any(f.name == "audio2.wav" for f in files)
    
    def test_discover_audio_files_non_recursive(self, tmp_path):
        """Test non-recursive audio file discovery."""
        # Create nested structure
        subdir = tmp_path / "subfolder"
        subdir.mkdir()
        
        (tmp_path / "audio1.mp3").touch()
        (subdir / "audio2.wav").touch()
        
        config = TranscriptionConfig(audio_folder=tmp_path, recursive=False)
        processor = AudioFileProcessor(config)
        files = processor.discover_audio_files()
        
        assert len(files) == 1
        assert files[0].name == "audio1.mp3"
    
    def test_discover_audio_files_no_source(self):
        """Test discovery with no audio source."""
        config = TranscriptionConfig()
        processor = AudioFileProcessor(config)
        
        with pytest.raises(ValueError, match="No audio source specified"):
            processor.discover_audio_files()
    
    def test_get_output_path_single_file_specified(self, config_with_file):
        """Test output path for single file with specified output."""
        config_with_file.transcription_file = Path("custom_output.md")
        processor = AudioFileProcessor(config_with_file)
        
        output_path = processor.get_output_path(config_with_file.audio_file, "md")
        assert output_path == Path("custom_output.md")
    
    def test_get_output_path_alongside_audio(self, config_with_file):
        """Test output path alongside audio file."""
        processor = AudioFileProcessor(config_with_file)
        
        output_path = processor.get_output_path(config_with_file.audio_file, "md")
        expected = config_with_file.audio_file.parent / "test_transcription.md"
        assert output_path == expected
    
    def test_get_output_path_custom_folder(self, config_with_file, tmp_path):
        """Test output path in custom folder."""
        config_with_file.transcribe_folder = tmp_path / "transcriptions"
        processor = AudioFileProcessor(config_with_file)
        
        output_path = processor.get_output_path(config_with_file.audio_file, "json")
        expected = tmp_path / "transcriptions" / "test_transcription.json"
        assert output_path == expected
    
    def test_get_extension_for_format(self, config_with_file):
        """Test extension mapping for different formats."""
        processor = AudioFileProcessor(config_with_file)
        
        assert processor._get_extension_for_format("md") == "md"
        assert processor._get_extension_for_format("txt") == "txt"
        assert processor._get_extension_for_format("json") == "json"
        assert processor._get_extension_for_format("srt") == "srt"
        assert processor._get_extension_for_format("vtt") == "vtt"
        assert processor._get_extension_for_format("unknown") == "txt"
    
    def test_batch_files(self, config_with_file):
        """Test file batching."""
        config_with_file.batch_size = 2
        processor = AudioFileProcessor(config_with_file)
        
        files = [Path(f"file{i}.mp3") for i in range(5)]
        batches = list(processor.batch_files(files))
        
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1
    
    def test_validate_output_path_success(self, config_with_file, tmp_path):
        """Test successful output path validation."""
        processor = AudioFileProcessor(config_with_file)
        output_path = tmp_path / "output.md"
        
        # Should not raise exception
        processor.validate_output_path(output_path)
        
        # Parent directory should exist
        assert output_path.parent.exists()
    
    def test_validate_output_path_creates_parent(self, config_with_file, tmp_path):
        """Test output path validation creates parent directories."""
        processor = AudioFileProcessor(config_with_file)
        output_path = tmp_path / "nested" / "folder" / "output.md"
        
        assert not output_path.parent.exists()
        processor.validate_output_path(output_path)
        assert output_path.parent.exists()
    
    def test_get_file_stats(self, config_with_file, tmp_path):
        """Test file statistics calculation."""
        # Create test files
        files = []
        for i, ext in enumerate(['.mp3', '.wav', '.m4a'], 1):
            file_path = tmp_path / f"audio{i}{ext}"
            file_path.write_bytes(b'0' * (1024 * i))  # Different sizes
            files.append(file_path)
        
        processor = AudioFileProcessor(config_with_file)
        stats = processor.get_file_stats(files)
        
        assert stats['total_files'] == 3
        assert stats['total_size_mb'] > 0
        assert stats['file_types'] == {'.mp3': 1, '.wav': 1, '.m4a': 1}
        assert stats['batch_count'] == 3  # batch_size = 1
    
    def test_is_audio_file(self, config_with_file):
        """Test audio file detection."""
        processor = AudioFileProcessor(config_with_file)
        
        assert processor._is_audio_file(Path("audio.mp3"))
        assert processor._is_audio_file(Path("audio.WAV"))
        assert processor._is_audio_file(Path("audio.M4A"))
        assert not processor._is_audio_file(Path("document.pdf"))
        assert not processor._is_audio_file(Path("image.jpg"))