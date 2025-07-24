import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import patch

from src.cli import main

class TestCLI:
    """Test cases for CLI interface."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()
    
    def test_models_info_flag(self, runner):
        """Test --models-info flag."""
        with patch('src.cli.TranscriptionApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            
            result = runner.invoke(main, ['--models-info'])
            
            assert result.exit_code == 0
            mock_app.show_model_info.assert_called_once()
    
    def test_single_file_transcription(self, runner, tmp_path):
        """Test transcribing a single file."""
        # Create test audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        with patch('src.cli.TranscriptionApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            
            result = runner.invoke(main, [
                '--audio-file', str(audio_file),
                '--model', 'base',
                '--output-format', 'txt',
                '--verbose'
            ])
            
            assert result.exit_code == 0
            mock_app.run.assert_called_once()
            
            # Verify config was created correctly
            config = mock_app_class.call_args[0][0]
            assert config.audio_file == audio_file
            assert config.model == 'base'
            assert config.output_format == 'txt'
            assert config.verbose is True
    
    def test_folder_transcription(self, runner, tmp_path):
        """Test transcribing files in a folder."""
        with patch('src.cli.TranscriptionApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            
            result = runner.invoke(main, [
                '--audio-folder', str(tmp_path),
                '--recursive',
                '--batch-size', '3',
                '--output-format', 'json'
            ])
            
            assert result.exit_code == 0
            mock_app.run.assert_called_once()
            
            # Verify config was created correctly
            config = mock_app_class.call_args[0][0]
            assert config.audio_folder == tmp_path
            assert config.recursive is True
            assert config.batch_size == 3
            assert config.output_format == 'json'
    
    def test_custom_whisper_parameters(self, runner, tmp_path):
        """Test custom Whisper parameters."""
        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        
        with patch('src.cli.TranscriptionApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            
            result = runner.invoke(main, [
                '--audio-file', str(audio_file),
                '--model', 'large',
                '--language', 'fr',
                '--device', 'cuda',
                '--temperature', '0.2',
                '--top-p', '0.9',
                '--top-k', '50',
                '--max-length', '1000'
            ])
            
            assert result.exit_code == 0
            
            # Verify config parameters
            config = mock_app_class.call_args[0][0]
            assert config.model == 'large'
            assert config.language == 'fr'
            assert config.device == 'cuda'
            assert config.temperature == 0.2
            assert config.top_p == 0.9
            assert config.top_k == 50
            assert config.max_length == 1000
    
    def test_output_path_options(self, runner, tmp_path):
        """Test output path configuration."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        transcribe_folder = tmp_path / "transcriptions"
        transcription_file = tmp_path / "custom_output.md"
        
        with patch('src.cli.TranscriptionApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            
            result = runner.invoke(main, [
                '--audio-file', str(audio_file),
                '--transcribe-folder', str(transcribe_folder),
                '--transcription-file', str(transcription_file)
            ])
            
            assert result.exit_code == 0
            
            # Verify config paths
            config = mock_app_class.call_args[0][0]
            assert config.transcribe_folder == transcribe_folder
            assert config.transcription_file == transcription_file
    
    def test_invalid_model_choice(self, runner, tmp_path):
        """Test invalid model choice."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        result = runner.invoke(main, [
            '--audio-file', str(audio_file),
            '--model', 'invalid_model'
        ])
        
        assert result.exit_code == 2  # Click error for invalid choice
        assert 'Invalid value' in result.output
    
    def test_invalid_output_format(self, runner, tmp_path):
        """Test invalid output format."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        result = runner.invoke(main, [
            '--audio-file', str(audio_file),
            '--output-format', 'invalid_format'
        ])
        
        assert result.exit_code == 2  # Click error for invalid choice
        assert 'Invalid value' in result.output
    
    def test_invalid_device(self, runner, tmp_path):
        """Test invalid device choice."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        result = runner.invoke(main, [
            '--audio-file', str(audio_file),
            '--device', 'invalid_device'
        ])
        
        assert result.exit_code == 2  # Click error for invalid choice
        assert 'Invalid value' in result.output
    
    def test_application_error_handling(self, runner, tmp_path):
        """Test error handling when application raises exception."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        with patch('src.cli.TranscriptionApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            mock_app.run.side_effect = Exception("Test error")
            
            result = runner.invoke(main, [
                '--audio-file', str(audio_file)
            ])
            
            assert result.exit_code == 1  # ClickException
            assert 'Error: Test error' in result.output
    
    def test_nonexistent_audio_file(self, runner):
        """Test with non-existent audio file."""
        result = runner.invoke(main, [
            '--audio-file', '/nonexistent/file.mp3'
        ])
        
        assert result.exit_code == 2  # Click error for invalid path
        assert 'does not exist' in result.output
    
    def test_nonexistent_audio_folder(self, runner):
        """Test with non-existent audio folder."""
        result = runner.invoke(main, [
            '--audio-folder', '/nonexistent/folder'
        ])
        
        assert result.exit_code == 2  # Click error for invalid path
        assert 'does not exist' in result.output
    
    def test_help_message(self, runner):
        """Test help message display."""
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert 'Transcribe audio files using OpenAI Whisper' in result.output
        assert '--audio-file' in result.output
        assert '--audio-folder' in result.output
        assert '--model' in result.output
        assert '--output-format' in result.output
        
        # Check examples are included
        assert 'Examples:' in result.output
        assert 'python -m src.cli' in result.output
    
    def test_default_values(self, runner, tmp_path):
        """Test default configuration values."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        with patch('src.cli.TranscriptionApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            
            result = runner.invoke(main, [
                '--audio-file', str(audio_file)
            ])
            
            assert result.exit_code == 0
            
            # Verify default values
            config = mock_app_class.call_args[0][0]
            assert config.model == 'base'
            assert config.language == 'en'
            assert config.output_format == 'md'
            assert config.device == 'cpu'
            assert config.temperature == 0.0
            assert config.top_p == 1.0
            assert config.top_k is None
            assert config.batch_size == 1
            assert config.max_length is None
            assert config.recursive is False
            assert config.verbose is False