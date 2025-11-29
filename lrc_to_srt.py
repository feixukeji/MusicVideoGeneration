import re

lrc_path = "./autodl-tmp/lyrics.lrc"
srt_path = "./autodl-tmp/lyrics.srt"
offset = 0  # Custom offset in seconds


def parse_lrc(lrc_path):
    """Parse LRC file and extract timestamps with lyrics."""
    with open(lrc_path, "r", encoding="utf-8") as f:
        lyrics = f.readlines()

    parsed_lyrics = []
    for line in lyrics:
        match = re.match(r'\[(\d+):(\d+\.\d+)\](.*)', line)
        if match:
            timestamp = int(match.group(1)) * 60 + float(match.group(2))
            content = match.group(3)
            parsed_lyrics.append((timestamp, content.strip()))
    return parsed_lyrics

def format_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

def lrc_to_srt(lrc_path, srt_path, offset=0.0):
    """Convert LRC file to SRT format."""
    parsed_lyrics = parse_lrc(lrc_path)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (timestamp, content) in enumerate(parsed_lyrics):
            start_time = format_srt_timestamp(timestamp + offset)
            end_time = format_srt_timestamp(parsed_lyrics[i + 1][0] + offset) if i + 1 < len(parsed_lyrics) else format_srt_timestamp(timestamp + offset + 5)
            f.write(f"{i + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{content}\n\n")

lrc_to_srt(lrc_path, srt_path, offset)
