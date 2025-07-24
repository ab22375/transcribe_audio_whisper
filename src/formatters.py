import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

class OutputFormatter:
    """Format transcription results for different output formats."""
    
    @staticmethod
    def format_markdown(result: Dict[str, Any]) -> str:
        """Format transcription result as Markdown."""
        metadata = result.get('metadata', {})
        text = result.get('text', '').strip()
        
        content = f"""# Transcription

**File:** {metadata.get('file_path', 'Unknown')}
**Model:** {metadata.get('model', 'Unknown')}
**Language:** {metadata.get('language', 'Unknown')}
**Duration:** {metadata.get('duration', 0):.2f} seconds
**Generated:** {datetime.now().isoformat()}

## Content

{text}
"""
        
        # Add segments if available
        segments = result.get('segments', [])
        if segments:
            content += "\n## Segments\n\n"
            for i, segment in enumerate(segments, 1):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                segment_text = segment.get('text', '').strip()
                content += f"**{i}.** [{start:.2f}s - {end:.2f}s] {segment_text}\n\n"
        
        return content
    
    @staticmethod
    def format_text(result: Dict[str, Any]) -> str:
        """Format transcription result as plain text."""
        text = result.get('text', '').strip()
        metadata = result.get('metadata', {})
        
        header = f"""Transcription of: {metadata.get('file_path', 'Unknown')}
Model: {metadata.get('model', 'Unknown')}
Language: {metadata.get('language', 'Unknown')}
Duration: {metadata.get('duration', 0):.2f} seconds
Generated: {datetime.now().isoformat()}

{'='*50}

"""
        return header + text
    
    @staticmethod
    def format_json(result: Dict[str, Any]) -> str:
        """Format transcription result as JSON."""
        # Add generation timestamp
        result['generated_at'] = datetime.now().isoformat()
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    @staticmethod
    def format_srt(result: Dict[str, Any]) -> str:
        """Format transcription result as SRT subtitle format."""
        segments = result.get('segments', [])
        if not segments:
            # If no segments, create a single subtitle from the full text
            text = result.get('text', '').strip()
            duration = result.get('metadata', {}).get('duration', 30)
            return f"""1
00:00:00,000 --> {OutputFormatter._format_srt_time(duration)}
{text}

"""
        
        srt_content = ""
        for i, segment in enumerate(segments, 1):
            start_time = OutputFormatter._format_srt_time(segment.get('start', 0))
            end_time = OutputFormatter._format_srt_time(segment.get('end', 0))
            text = segment.get('text', '').strip()
            
            srt_content += f"""{i}
{start_time} --> {end_time}
{text}

"""
        
        return srt_content
    
    @staticmethod
    def format_vtt(result: Dict[str, Any]) -> str:
        """Format transcription result as WebVTT format."""
        segments = result.get('segments', [])
        
        vtt_content = "WEBVTT\n\n"
        
        if not segments:
            # If no segments, create a single subtitle from the full text
            text = result.get('text', '').strip()
            duration = result.get('metadata', {}).get('duration', 30)
            vtt_content += f"""00:00:00.000 --> {OutputFormatter._format_vtt_time(duration)}
{text}

"""
            return vtt_content
        
        for segment in segments:
            start_time = OutputFormatter._format_vtt_time(segment.get('start', 0))
            end_time = OutputFormatter._format_vtt_time(segment.get('end', 0))
            text = segment.get('text', '').strip()
            
            vtt_content += f"""{start_time} --> {end_time}
{text}

"""
        
        return vtt_content
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """Format seconds to WebVTT time format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    @classmethod
    def format_result(cls, result: Dict[str, Any], output_format: str) -> str:
        """Format result according to specified format."""
        formatters = {
            'md': cls.format_markdown,
            'txt': cls.format_text,
            'json': cls.format_json,
            'srt': cls.format_srt,
            'vtt': cls.format_vtt
        }
        
        formatter = formatters.get(output_format)
        if not formatter:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return formatter(result)
    
    @staticmethod
    def save_result(content: str, output_path: Path) -> None:
        """Save formatted content to file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding='utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to save to {output_path}: {e}")