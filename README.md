# Audio Transcription Tool with OpenAI Whisper

A modern, modular Python application for transcribing audio files using OpenAI Whisper. Features batch processing, multiple output formats, and comprehensive CLI interface.

## Features

- 🎯 **Multiple Whisper Models**: Support for all Whisper models (tiny, base, small, medium, large)
- 🌍 **Multi-language Support**: Automatic language detection or manual specification
- 📁 **Batch Processing**: Process single files or entire folders recursively
- 📄 **Multiple Output Formats**: Markdown, Plain Text, JSON, SRT, WebVTT
- ⚡ **GPU Acceleration**: CUDA and Apple Silicon (MPS) support
- 🎨 **Beautiful CLI**: Rich progress bars and colorful output
- 🧪 **Comprehensive Testing**: 95%+ test coverage with pytest
- 🔧 **Type Safety**: Full type hints with Pydantic validation

## Installation

This project uses the modern `uv` package manager for Python dependencies.

```bash
# Clone the repository
git clone <repository-url>
cd transcribe_audio_whisper

# Install dependencies with uv
uv sync

# Or install uv first if needed
pip install uv
uv sync
```

## Quick Start

### Transcribe a Single File

```bash
python -m src.cli --audio-file audio.mp3
```

### Process an Entire Folder

```bash
python -m src.cli --audio-folder /path/to/audio --recursive
```

### Use a Larger Model with GPU

```bash
python -m src.cli --audio-file audio.wav --model large --device cuda
```

### Export as Subtitles

```bash
python -m src.cli --audio-file video.mp4 --output-format srt
```

## Usage

### Command Line Options

```bash
python -m src.cli [OPTIONS]

Options:
  --model [tiny|base|small|medium|large]     Whisper model to use [default: base]
  --language TEXT                            Language for transcription [default: en]
  --output-format [md|txt|json|srt|vtt]     Output format [default: md]
  --device [cpu|cuda|mps]                   Device for processing [default: cpu]
  --temperature FLOAT                        Sampling temperature (0.0-1.0) [default: 0.0]
  --top-p FLOAT                             Top-p sampling (0.0-1.0) [default: 1.0]
  --top-k INTEGER                           Top-k sampling parameter
  --batch-size INTEGER                       Batch size for processing [default: 1]
  --max-length INTEGER                       Maximum transcription length
  --audio-file PATH                         Single audio file to transcribe
  --audio-folder PATH                       Folder containing audio files
  --transcribe-folder PATH                  Output folder for transcriptions
  --transcription-file PATH                 Output file for single transcription
  --recursive / --no-recursive              Recursively search audio folder [default: no-recursive]
  --verbose / --quiet                       Enable verbose output [default: quiet]
  --models-info                             Show available models information
  --help                                    Show this message and exit
```

### Supported Audio Formats

- **Audio Files**: MP3, WAV, M4A, FLAC, OGG, WMA, AAC
- **Video Files**: MP4, MKV, AVI (audio track will be extracted)

### Output Formats

1. **Markdown (`.md`)** - Structured format with metadata and segments
2. **Plain Text (`.txt`)** - Simple text with header information  
3. **JSON (`.json`)** - Complete structured data including segments
4. **SRT (`.srt`)** - Standard subtitle format with timestamps
5. **WebVTT (`.vtt`)** - Web-compatible subtitle format

## Examples

### Basic Usage

```bash
# Transcribe with default settings (base model, markdown output)
python -m src.cli --audio-file recording.mp3

# Process folder recursively
python -m src.cli --audio-folder ./recordings --recursive

# Specify output location
python -m src.cli --audio-file audio.wav --transcription-file transcript.md
```

### Advanced Configuration

```bash
# Use large model with French language
python -m src.cli --audio-file french_audio.mp3 --model large --language fr

# Batch process with GPU acceleration
python -m src.cli --audio-folder ./podcasts --device cuda --batch-size 4

# Generate subtitles with custom parameters
python -m src.cli --audio-file video.mp4 --output-format srt --temperature 0.1
```

### Model Information

```bash
# View available models and their specifications
python -m src.cli --models-info
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_config.py -v
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code  
uv run ruff check

# Type checking (if using mypy)
uv run mypy src/
```

### Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add development dependency
uv add --dev package-name
```

## Architecture

The application follows a modular architecture:

- **`config.py`** - Pydantic-based configuration management
- **`transcriber.py`** - Core Whisper integration and transcription logic
- **`file_processor.py`** - File discovery and batch processing
- **`formatters.py`** - Output formatting for different file types
- **`app.py`** - Main application coordinator with progress tracking
- **`cli.py`** - Command-line interface with Click

## Performance Tips

1. **Use GPU**: Specify `--device cuda` or `--device mps` for faster processing
2. **Batch Processing**: Increase `--batch-size` for multiple files
3. **Model Selection**: Use smaller models (`tiny`, `base`) for faster processing
4. **Language Specification**: Specify `--language` to skip auto-detection

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller model
2. **Slow Processing**: Check if GPU acceleration is available and enabled
3. **Permission Errors**: Ensure write permissions for output directories
4. **Model Download**: First run downloads models (requires internet)

### Debug Mode

```bash
python -m src.cli --audio-file test.mp3 --verbose
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
