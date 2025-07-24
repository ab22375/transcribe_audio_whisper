import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.config import TranscriptionConfig
from src.transcriber import WhisperTranscriber

class TestWhisperTranscriber:
    """Test cases for WhisperTranscriber."""
    
    @pytest.fixture
    def config(self):
        """Create basic config for testing."""
        return TranscriptionConfig(
            model="base",
            language="en",
            audio_file=Path("test.mp3")
        )
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model."""
        mock_model = Mock()
        mock_model.is_multilingual = True
        mock_model.parameters.return_value = [Mock()]
        mock_model.parameters.return_value[0].device = "cpu"
        mock_model.parameters.return_value[0].numel.return_value = 1000000
        return mock_model
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_init_loads_model(self, mock_torch, mock_load_model, config, mock_whisper_model):
        """Test that initialization loads the Whisper model."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_load_model.return_value = mock_whisper_model
        
        transcriber = WhisperTranscriber(config)
        
        mock_load_model.assert_called_once_with("base", device="cpu")
        assert transcriber.model == mock_whisper_model
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_get_device_cuda(self, mock_torch, mock_load_model, mock_whisper_model):
        """Test device selection with CUDA."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False
        mock_load_model.return_value = mock_whisper_model
        
        config = TranscriptionConfig(device="cuda", audio_file=Path("test.mp3"))
        transcriber = WhisperTranscriber(config)
        
        mock_load_model.assert_called_once_with("base", device="cuda")
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_get_device_mps(self, mock_torch, mock_load_model, mock_whisper_model):
        """Test device selection with MPS."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_load_model.return_value = mock_whisper_model
        
        config = TranscriptionConfig(device="mps", audio_file=Path("test.mp3"))
        transcriber = WhisperTranscriber(config)
        
        mock_load_model.assert_called_once_with("base", device="mps")
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_get_device_fallback_to_cpu(self, mock_torch, mock_load_model, mock_whisper_model):
        """Test device fallback to CPU when CUDA/MPS unavailable."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_load_model.return_value = mock_whisper_model
        
        config = TranscriptionConfig(device="cuda", audio_file=Path("test.mp3"))
        transcriber = WhisperTranscriber(config)
        
        mock_load_model.assert_called_once_with("base", device="cpu")
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_transcribe_file_success(self, mock_torch, mock_load_model, config, mock_whisper_model, tmp_path):
        """Test successful file transcription."""
        # Setup
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_load_model.return_value = mock_whisper_model
        
        # Create test audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        # Mock transcription result
        mock_result = {
            'text': 'Hello world',
            'language': 'en',
            'segments': [
                {'start': 0.0, 'end': 1.0, 'text': 'Hello'},
                {'start': 1.0, 'end': 2.0, 'text': 'world'}
            ]
        }
        mock_whisper_model.transcribe.return_value = mock_result
        
        transcriber = WhisperTranscriber(config)
        result = transcriber.transcribe_file(audio_file)
        
        # Verify transcription was called
        mock_whisper_model.transcribe.assert_called_once_with(
            str(audio_file),
            language='en'
        )
        
        # Verify result structure
        assert result['text'] == 'Hello world'
        assert result['language'] == 'en'
        assert 'metadata' in result
        assert result['metadata']['file_path'] == str(audio_file)
        assert result['metadata']['model'] == 'base'
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_transcribe_file_not_found(self, mock_torch, mock_load_model, config, mock_whisper_model):
        """Test transcription of non-existent file."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_load_model.return_value = mock_whisper_model
        
        transcriber = WhisperTranscriber(config)
        
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcriber.transcribe_file(Path("nonexistent.mp3"))
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_transcribe_unsupported_format(self, mock_torch, mock_load_model, config, mock_whisper_model, tmp_path):
        """Test transcription of unsupported file format."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_load_model.return_value = mock_whisper_model
        
        # Create unsupported file
        unsupported_file = tmp_path / "test.pdf"
        unsupported_file.touch()
        
        transcriber = WhisperTranscriber(config)
        
        with pytest.raises(ValueError, match="Unsupported audio format"):
            transcriber.transcribe_file(unsupported_file)
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_get_transcription_options_default(self, mock_torch, mock_load_model, config, mock_whisper_model):
        """Test default transcription options."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_load_model.return_value = mock_whisper_model
        
        transcriber = WhisperTranscriber(config)
        options = transcriber._get_transcription_options()
        
        assert options['language'] == 'en'
        # Temperature is 0.0 (default), should not be in options
        assert 'temperature' not in options
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_get_transcription_options_custom(self, mock_torch, mock_load_model, mock_whisper_model):
        """Test custom transcription options."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_load_model.return_value = mock_whisper_model
        
        config = TranscriptionConfig(
            audio_file=Path("test.mp3"),
            language="fr",
            temperature=0.5,
            top_p=0.8,
            top_k=50
        )
        
        transcriber = WhisperTranscriber(config)
        options = transcriber._get_transcription_options()
        
        assert options['language'] == 'fr'
        assert options['temperature'] == 0.5
        assert options['decode_options']['top_p'] == 0.8
        assert options['decode_options']['top_k'] == 50
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_is_supported_format(self, mock_torch, mock_load_model, config, mock_whisper_model):
        """Test supported format detection."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_load_model.return_value = mock_whisper_model
        
        transcriber = WhisperTranscriber(config)
        
        # Test supported formats
        assert transcriber._is_supported_format(Path("audio.mp3"))
        assert transcriber._is_supported_format(Path("audio.WAV"))
        assert transcriber._is_supported_format(Path("audio.M4A"))
        
        # Test unsupported formats
        assert not transcriber._is_supported_format(Path("document.pdf"))
        assert not transcriber._is_supported_format(Path("image.jpg"))
    
    @patch('src.transcriber.whisper.load_model')
    @patch('src.transcriber.torch')
    def test_get_model_info(self, mock_torch, mock_load_model, config, mock_whisper_model):
        """Test model information retrieval."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_load_model.return_value = mock_whisper_model
        
        # Mock parameter device and count
        mock_param = Mock()
        mock_param.device = "cpu"
        mock_param.numel.return_value = 1000000
        mock_whisper_model.parameters.return_value = [mock_param]
        
        transcriber = WhisperTranscriber(config)
        info = transcriber.get_model_info()
        
        assert info['model_name'] == 'base'
        assert info['device'] == 'cpu'
        assert info['parameters'] == 1000000
        assert info['is_multilingual'] is True
    
    @patch('src.transcriber.whisper.load_model')
    def test_model_loading_failure(self, mock_load_model, config):
        """Test model loading failure."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            WhisperTranscriber(config)