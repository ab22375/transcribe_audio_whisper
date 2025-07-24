from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from .config import TranscriptionConfig
from .transcriber import WhisperTranscriber
from .file_processor import AudioFileProcessor
from .formatters import OutputFormatter

class TranscriptionApp:
    """Main application for audio transcription."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.console = Console()
        self.transcriber = None
        self.file_processor = AudioFileProcessor(config)
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logger.remove()  # Remove default handler
        
        if self.config.verbose:
            logger.add(
                lambda msg: self.console.print(f"[dim]{msg}[/dim]", markup=False),
                level="DEBUG",
                format="{time:HH:mm:ss} | {level} | {message}"
            )
        else:
            logger.add(
                lambda msg: self.console.print(f"[red]ERROR:[/red] {msg}", markup=False) if "ERROR" in str(msg) else None,
                level="ERROR",
                format="{message}"
            )
    
    def run(self) -> None:
        """Run the transcription process."""
        try:
            # Validate configuration
            self.config.validate_paths()
            
            # Initialize transcriber
            self.transcriber = WhisperTranscriber(self.config)
            
            # Discover audio files
            audio_files = self.file_processor.discover_audio_files()
            
            if not audio_files:
                self.console.print("[yellow]No audio files found to transcribe.[/yellow]")
                return
            
            # Show processing info
            self._show_processing_info(audio_files)
            
            # Process files
            self._process_files(audio_files)
            
            self.console.print("\n[green]✓ Transcription completed successfully![/green]")
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            self.console.print(f"[red]Error: {e}[/red]")
            raise
    
    def _show_processing_info(self, files: List[Path]) -> None:
        """Display information about files to be processed."""
        if not self.config.verbose:
            return
        
        stats = self.file_processor.get_file_stats(files)
        model_info = self.transcriber.get_model_info()
        
        # Create info table
        table = Table(title="Transcription Setup", show_header=False, box=None)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Model", model_info.get('model_name', 'Unknown'))
        table.add_row("Device", model_info.get('device', 'Unknown'))
        table.add_row("Language", self.config.language or "Auto-detect")
        table.add_row("Output Format", self.config.output_format.upper())
        table.add_row("Files to Process", str(stats['total_files']))
        table.add_row("Total Size", f"{stats['total_size_mb']} MB")
        table.add_row("Batch Size", str(self.config.batch_size))
        
        self.console.print()
        self.console.print(table)
        self.console.print()
    
    def _process_files(self, files: List[Path]) -> None:
        """Process all audio files with progress tracking."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            main_task = progress.add_task("Processing files...", total=len(files))
            
            for batch in self.file_processor.batch_files(files):
                batch_task = progress.add_task(
                    f"Batch ({len(batch)} files)", 
                    total=len(batch)
                )
                
                for audio_file in batch:
                    progress.update(
                        batch_task, 
                        description=f"Transcribing {audio_file.name}"
                    )
                    
                    try:
                        self._process_single_file(audio_file)
                        progress.advance(batch_task)
                        progress.advance(main_task)
                        
                    except Exception as e:
                        logger.error(f"Failed to process {audio_file}: {e}")
                        progress.advance(batch_task)
                        progress.advance(main_task)
                        continue
                
                progress.remove_task(batch_task)
    
    def _process_single_file(self, audio_file: Path) -> None:
        """Process a single audio file."""
        # Transcribe
        result = self.transcriber.transcribe_file(audio_file)
        
        # Get output path
        output_path = self.file_processor.get_output_path(
            audio_file, 
            self.config.output_format
        )
        
        # Validate output path
        self.file_processor.validate_output_path(output_path)
        
        # Format and save
        formatted_content = OutputFormatter.format_result(
            result, 
            self.config.output_format
        )
        
        OutputFormatter.save_result(formatted_content, output_path)
        
        if self.config.verbose:
            self.console.print(f"[green]✓[/green] Saved transcription: {output_path}")
    
    def show_model_info(self) -> None:
        """Display information about available models."""
        available_models = [
            "tiny", "tiny.en", "base", "base.en", 
            "small", "small.en", "medium", "medium.en", 
            "large-v1", "large-v2", "large-v3", "large"
        ]
        
        table = Table(title="Available Whisper Models")
        table.add_column("Model", style="cyan")
        table.add_column("Languages", style="white")
        table.add_column("Size", style="yellow")
        table.add_column("Speed", style="green")
        
        model_info = [
            ("tiny", "Multilingual", "~39 MB", "Very Fast"),
            ("tiny.en", "English only", "~39 MB", "Very Fast"),
            ("base", "Multilingual", "~74 MB", "Fast"),
            ("base.en", "English only", "~74 MB", "Fast"),
            ("small", "Multilingual", "~244 MB", "Medium"),
            ("small.en", "English only", "~244 MB", "Medium"),
            ("medium", "Multilingual", "~769 MB", "Slow"),
            ("medium.en", "English only", "~769 MB", "Slow"),
            ("large-v1", "Multilingual", "~1550 MB", "Very Slow"),
            ("large-v2", "Multilingual", "~1550 MB", "Very Slow"),
            ("large-v3", "Multilingual", "~1550 MB", "Very Slow"),
            ("large", "Multilingual", "~1550 MB", "Very Slow"),
        ]
        
        for model, langs, size, speed in model_info:
            table.add_row(model, langs, size, speed)
        
        self.console.print(table)