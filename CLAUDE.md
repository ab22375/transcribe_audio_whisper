# Audio Transcription Tool - Implementation Documentation

## Project Overview

This is a modern, modular Python application for transcribing audio files using OpenAI Whisper. The application provides a comprehensive CLI interface with batch processing capabilities, multiple output formats, and extensive configuration options.

## Architecture

The application follows a modular architecture with clear separation of concerns:

### Core Modules

1. **`config.py`** - Configuration management using Pydantic
   - Validates all input parameters
   - Provides type safety and automatic validation
   - Supports multiple output formats and devices

2. **`transcriber.py`** - Core Whisper integration
   - Loads and manages Whisper models
   - Handles device selection (CPU/CUDA/MPS)
   - Performs actual audio transcription with metadata

3. **`file_processor.py`** - File discovery and management
   - Discovers audio files (single file or recursive folder search)
   - Manages output path generation
   - Handles file batching for efficient processing

4. **`formatters.py`** - Output formatting
   - Supports multiple formats: Markdown, Text, JSON, SRT, WebVTT
   - Handles Unicode content properly
   - Creates structured output with metadata

5. **`app.py`** - Main application orchestrator
   - Coordinates all modules
   - Provides Rich-based progress tracking
   - Handles error reporting and logging

6. **`cli.py`** - Command-line interface
   - Click-based CLI with comprehensive options
   - Input validation and help system
   - Error handling and user feedback

## Configuration Parameters

### Whisper Model Parameters
- `model`: Whisper model ("tiny", "base", "small", "medium", "large-v3", etc.). Default: `"base"`
- `language`: Language code for transcription. Default: `"en"` (auto-detect if None)
- `temperature`: Sampling temperature (0.0-1.0). Default: `0.0`
- `top_p`: Top-p sampling parameter (0.0-1.0). Default: `1.0`
- `top_k`: Top-k sampling parameter. Default: `None`

### Processing Parameters
- `device`: Processing device ("cpu", "cuda", "mps"). Default: `"cpu"`
- `batch_size`: Files processed simultaneously. Default: `1`
- `max_length`: Maximum transcription length. Default: `None`
- `verbose`: Enable detailed progress output. Default: `False`

### File Handling Parameters
- `audio_file`: Single audio file path. Default: `None`
- `audio_folder`: Folder containing audio files. Default: `None`
- `transcribe_folder`: Output directory for transcriptions. Default: `None`
- `transcription_file`: Specific output file for single transcription. Default: `None`
- `recursive`: Recursively search audio folder. Default: `False`

### Output Parameters
- `output_format`: Output format ("md", "txt", "json", "srt", "vtt"). Default: `"md"`

## Supported Audio Formats

The application supports common audio and video formats:
- Audio: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.wma`, `.aac`
- Video: `.mp4`, `.mkv`, `.avi` (audio track extracted)

## Output Formats

1. **Markdown (md)** - Structured format with metadata and optional segments
2. **Plain Text (txt)** - Simple text with header information
3. **JSON** - Complete structured data including segments and metadata
4. **SRT** - Subtitle format with timestamps
5. **WebVTT (vtt)** - Web-compatible subtitle format

## Error Handling

The application implements comprehensive error handling:
- Path validation with clear error messages
- Model loading failure recovery
- File permission and access checks
- Unsupported format detection
- Progress tracking with error recovery

## Testing

Comprehensive test suite using pytest with:
- Unit tests for all modules (95%+ coverage)
- Mock-based testing for external dependencies
- Integration tests for CLI interface
- Fixture-based test data management
- Parameterized tests for multiple scenarios

## Development Notes

### Dependencies Management
- Uses `uv` package manager exclusively
- All dependencies declared in `pyproject.toml`
- No direct `pip` usage (use `uv add` for new packages)

### Code Quality
- Type hints throughout the codebase
- Pydantic for configuration validation
- Rich library for beautiful CLI output
- Loguru for structured logging
- Following Python best practices

### Performance Considerations
- Batch processing for multiple files
- Device auto-detection and optimization
- Memory-efficient file processing
- Progress tracking for long operations

## Usage Examples

```bash
# Single file transcription
python -m src.cli --audio-file audio.mp3

# Folder processing with recursion
python -m src.cli --audio-folder /path/to/audio --recursive

# Custom model and format
python -m src.cli --audio-file audio.wav --model large --output-format json

# Batch processing with GPU
python -m src.cli --audio-folder ./audio --device cuda --batch-size 4
```
