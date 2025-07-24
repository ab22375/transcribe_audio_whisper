import pytest
import json
from pathlib import Path
from datetime import datetime

from src.formatters import OutputFormatter

class TestOutputFormatter:
    """Test cases for OutputFormatter."""
    
    @pytest.fixture
    def sample_result(self):
        """Sample transcription result for testing."""
        return {
            'text': 'Hello, this is a test transcription.',
            'language': 'en',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.5,
                    'text': 'Hello, this is a test'
                },
                {
                    'start': 2.5,
                    'end': 5.0,
                    'text': ' transcription.'
                }
            ],
            'metadata': {
                'file_path': '/path/to/audio.mp3',
                'file_size': 1024000,
                'model': 'base',
                'language': 'en',
                'duration': 5.0
            }
        }
    
    def test_format_markdown(self, sample_result):
        """Test markdown formatting."""
        content = OutputFormatter.format_markdown(sample_result)
        
        assert '# Transcription' in content
        assert '**File:** /path/to/audio.mp3' in content
        assert '**Model:** base' in content
        assert '**Language:** en' in content
        assert '**Duration:** 5.00 seconds' in content
        assert '## Content' in content
        assert 'Hello, this is a test transcription.' in content
        assert '## Segments' in content
        assert '[0.00s - 2.50s]' in content
        assert '[2.50s - 5.00s]' in content
    
    def test_format_text(self, sample_result):
        """Test plain text formatting."""
        content = OutputFormatter.format_text(sample_result)
        
        assert 'Transcription of: /path/to/audio.mp3' in content
        assert 'Model: base' in content
        assert 'Language: en' in content
        assert 'Duration: 5.00 seconds' in content
        assert '=' * 50 in content
        assert 'Hello, this is a test transcription.' in content
    
    def test_format_json(self, sample_result):
        """Test JSON formatting."""
        content = OutputFormatter.format_json(sample_result)
        
        # Parse JSON to verify it's valid
        data = json.loads(content)
        
        assert data['text'] == sample_result['text']
        assert data['language'] == sample_result['language']
        assert data['segments'] == sample_result['segments']
        assert data['metadata'] == sample_result['metadata']
        assert 'generated_at' in data
        
        # Verify timestamp format
        datetime.fromisoformat(data['generated_at'])
    
    def test_format_srt(self, sample_result):
        """Test SRT subtitle formatting."""
        content = OutputFormatter.format_srt(sample_result)
        
        lines = content.strip().split('\n')
        
        # Check first subtitle
        assert lines[0] == '1'
        assert '00:00:00,000 --> 00:00:02,500' in lines[1]
        assert 'Hello, this is a test' in lines[2]
        
        # Check second subtitle
        assert lines[4] == '2'
        assert '00:00:02,500 --> 00:00:05,000' in lines[5]
        assert ' transcription.' in lines[6]
    
    def test_format_srt_no_segments(self):
        """Test SRT formatting with no segments."""
        result = {
            'text': 'Single line transcription',
            'metadata': {'duration': 10.0}
        }
        
        content = OutputFormatter.format_srt(result)
        lines = content.strip().split('\n')
        
        assert lines[0] == '1'
        assert '00:00:00,000 --> 00:00:10,000' in lines[1]
        assert 'Single line transcription' in lines[2]
    
    def test_format_vtt(self, sample_result):
        """Test WebVTT formatting."""
        content = OutputFormatter.format_vtt(sample_result)
        
        lines = content.strip().split('\n')
        
        assert lines[0] == 'WEBVTT'
        assert '00:00:00.000 --> 00:00:02.500' in content
        assert '00:00:02.500 --> 00:00:05.000' in content
        assert 'Hello, this is a test' in content
        assert ' transcription.' in content
    
    def test_format_vtt_no_segments(self):
        """Test WebVTT formatting with no segments."""
        result = {
            'text': 'Single line transcription',
            'metadata': {'duration': 15.5}
        }
        
        content = OutputFormatter.format_vtt(result)
        
        assert 'WEBVTT' in content
        assert '00:00:00.000 --> 00:00:15.500' in content
        assert 'Single line transcription' in content
    
    def test_srt_time_formatting(self):
        """Test SRT time formatting."""
        # Test various time values
        assert OutputFormatter._format_srt_time(0.0) == "00:00:00,000"
        assert OutputFormatter._format_srt_time(1.5) == "00:00:01,500"
        assert OutputFormatter._format_srt_time(61.123) == "00:01:01,123"
        assert OutputFormatter._format_srt_time(3661.999) == "01:01:01,999"
    
    def test_vtt_time_formatting(self):
        """Test WebVTT time formatting."""
        # Test various time values
        assert OutputFormatter._format_vtt_time(0.0) == "00:00:00.000"
        assert OutputFormatter._format_vtt_time(1.5) == "00:00:01.500"
        assert OutputFormatter._format_vtt_time(61.123) == "00:01:01.123"
        assert OutputFormatter._format_vtt_time(3661.999) == "01:01:01.999"
    
    def test_format_result_all_formats(self, sample_result):
        """Test format_result method with all formats."""
        formats = ['md', 'txt', 'json', 'srt', 'vtt']
        
        for fmt in formats:
            content = OutputFormatter.format_result(sample_result, fmt)
            assert isinstance(content, str)
            assert len(content) > 0
    
    def test_format_result_invalid_format(self, sample_result):
        """Test format_result with invalid format."""
        with pytest.raises(ValueError, match="Unsupported output format: invalid"):
            OutputFormatter.format_result(sample_result, 'invalid')
    
    def test_save_result_success(self, tmp_path):
        """Test successful result saving."""
        content = "Test transcription content"
        output_path = tmp_path / "test_output.txt"
        
        OutputFormatter.save_result(content, output_path)
        
        assert output_path.exists()
        assert output_path.read_text(encoding='utf-8') == content
    
    def test_save_result_creates_directories(self, tmp_path):
        """Test that save_result creates parent directories."""
        content = "Test transcription content"
        output_path = tmp_path / "nested" / "folder" / "test_output.txt"
        
        assert not output_path.parent.exists()
        
        OutputFormatter.save_result(content, output_path)
        
        assert output_path.exists()
        assert output_path.read_text(encoding='utf-8') == content
    
    def test_save_result_unicode_content(self, tmp_path):
        """Test saving Unicode content."""
        content = "Transcription with Unicode: 你好 🎵 café"
        output_path = tmp_path / "unicode_test.txt"
        
        OutputFormatter.save_result(content, output_path)
        
        assert output_path.exists()
        saved_content = output_path.read_text(encoding='utf-8')
        assert saved_content == content
    
    def test_markdown_format_without_segments(self):
        """Test markdown formatting without segments."""
        result = {
            'text': 'Simple transcription',
            'metadata': {
                'file_path': '/test.mp3',
                'model': 'base',
                'language': 'en',
                'duration': 3.0
            }
        }
        
        content = OutputFormatter.format_markdown(result)
        
        assert '# Transcription' in content
        assert '## Content' in content
        assert 'Simple transcription' in content
        # Should not have segments section
        assert '## Segments' not in content
    
    def test_formatting_handles_missing_metadata(self):
        """Test formatting handles missing metadata gracefully."""
        result = {
            'text': 'Test transcription'
        }
        
        # Should not raise exceptions
        md_content = OutputFormatter.format_markdown(result)
        txt_content = OutputFormatter.format_text(result)
        json_content = OutputFormatter.format_json(result)
        
        assert 'Test transcription' in md_content
        assert 'Test transcription' in txt_content
        
        json_data = json.loads(json_content)
        assert json_data['text'] == 'Test transcription'