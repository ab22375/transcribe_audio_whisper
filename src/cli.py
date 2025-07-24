from pathlib import Path
from typing import Optional
import click
from rich.console import Console

from .config import TranscriptionConfig, OutputFormat, Device
from .app import TranscriptionApp

console = Console()

@click.command()
@click.option('--model', default='base', help='Whisper model to use', 
              type=click.Choice(['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 
                               'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3', 'large']))
@click.option('--language', default='en', help='Language for transcription (auto-detect if not specified)')
@click.option('--output-format', default='md', type=click.Choice(['md', 'txt', 'json', 'srt', 'vtt']),
              help='Output format')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda', 'mps']),
              help='Device to use for processing')
@click.option('--temperature', default=0.0, type=float,
              help='Sampling temperature (0.0 to 1.0)')
@click.option('--top-p', default=1.0, type=float,
              help='Top-p sampling parameter (0.0 to 1.0)')
@click.option('--top-k', type=int,
              help='Top-k sampling parameter')
@click.option('--batch-size', default=1, type=int,
              help='Batch size for processing')
@click.option('--max-length', type=int,
              help='Maximum transcription length')
@click.option('--audio-file', type=click.Path(exists=True, path_type=Path),
              help='Single audio file to transcribe')
@click.option('--audio-folder', type=click.Path(exists=True, file_okay=False, path_type=Path),
              help='Folder containing audio files')
@click.option('--transcribe-folder', type=click.Path(path_type=Path),
              help='Output folder for transcriptions')
@click.option('--transcription-file', type=click.Path(path_type=Path),
              help='Output file for single transcription')
@click.option('--recursive/--no-recursive', default=False,
              help='Recursively search audio folder')
@click.option('--verbose/--quiet', default=False,
              help='Enable verbose output')
@click.option('--models-info', is_flag=True,
              help='Show information about available models')
def main(
    model: str,
    language: Optional[str],
    output_format: OutputFormat,
    device: Device,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    batch_size: int,
    max_length: Optional[int],
    audio_file: Optional[Path],
    audio_folder: Optional[Path],
    transcribe_folder: Optional[Path],
    transcription_file: Optional[Path],
    recursive: bool,
    verbose: bool,
    models_info: bool
):
    """Transcribe audio files using OpenAI Whisper.
    
    Examples:
    
        # Transcribe a single file
        python -m src.cli --audio-file audio.mp3
        
        # Transcribe all files in a folder
        python -m src.cli --audio-folder /path/to/audio --recursive
        
        # Use different model and output format
        python -m src.cli --audio-file audio.wav --model large --output-format json
        
        # Specify output location
        python -m src.cli --audio-file audio.mp3 --transcription-file transcript.md
    """
    
    if models_info:
        app = TranscriptionApp(TranscriptionConfig())
        app.show_model_info()
        return
    
    try:
        # Create configuration
        config = TranscriptionConfig(
            model=model,
            language=language,
            output_format=output_format,
            device=device,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            batch_size=batch_size,
            max_length=max_length,
            audio_file=audio_file,
            audio_folder=audio_folder,
            transcribe_folder=transcribe_folder,
            transcription_file=transcription_file,
            recursive=recursive,
            verbose=verbose
        )
        
        # Create and run application
        app = TranscriptionApp(config)
        app.run()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    main()