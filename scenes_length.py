import os
import subprocess
import json

scenes_dir = "./autodl-tmp/scenes/"
scenes_length_path = "./autodl-tmp/scenes_length.json"
scenes_length = {}

# Get the list of scene files and sort them by file name
scene_files = sorted(os.listdir(scenes_dir))

for scene_name in scene_files:
    scene_path = os.path.join(scenes_dir, scene_name)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        scene_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    scenes_length[scene_name] = float(result.stdout)
    print(scene_name, scenes_length[scene_name])

with open(scenes_length_path, "w", encoding="utf-8") as f:
    json.dump(scenes_length, f, ensure_ascii=False, indent=4)

print("Scenes length saved to", scenes_length_path)