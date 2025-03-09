import os
import json
import subprocess
import shutil

best_matches_path = "./autodl-tmp/best_matches.json"
scenes_dir = "./autodl-tmp/scenes/"
output_video_path = "./autodl-tmp/output_video.mp4"
tmp_dir = "./autodl-tmp/tmp_segments/"
tmp_list_path = "./tmp_segments_list.txt"

# Create temporary directory for video segments
os.makedirs(tmp_dir, exist_ok=True)

# Read best match results
with open(best_matches_path, "r", encoding="utf-8") as f:
    best_matches = json.load(f)

# Get all timestamps
timestamps = sorted(best_matches.keys(), key=lambda x: float(x))

# Process each timestamp and create temporary video segments
segment_files = []
for i, timestamp in enumerate(timestamps):
    filename = best_matches[timestamp]
    if filename == "None":
        continue

    scene_path = os.path.join(scenes_dir, filename)
    segment_path = os.path.join(tmp_dir, f"segment_{i}.mp4")

    # Calculate duration
    if i < len(timestamps) - 1:
        next_timestamp = float(timestamps[i + 1])
        duration = next_timestamp - float(timestamp)

        # Create a temporary segment with the specified duration
        subprocess.run([
            "ffmpeg", "-y", "-i", scene_path, "-t", str(duration), "-c", "copy", segment_path
        ])
    else:
        # If it's the last segment, just copy the file
        shutil.copy(scene_path, segment_path)

    segment_files.append(segment_path)

# Concatenate video segments using ffmpeg
with open(tmp_list_path, "w", encoding="utf-8") as f:
    for segment_file in segment_files:
        f.write(f"file '{segment_file}'\n")

command = [
    "ffmpeg",
    "-f", "concat",
    "-safe", "0",
    "-i", tmp_list_path,
    "-c", "copy",
    output_video_path
]

subprocess.run(command)

# Clean up temporary files
shutil.rmtree(tmp_dir)
os.remove(tmp_list_path)
