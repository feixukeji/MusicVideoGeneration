import os
import torch
import warnings
from transnetv2_pytorch import TransNetV2
import ffmpeg
import json
import argparse

# Filter out warnings related to deterministic algorithm
warnings.filterwarnings("ignore", message=".*Deterministic behavior was enabled.*")

# Argument parser for optional deletion of source videos
parser = argparse.ArgumentParser()
parser.add_argument("--delete-source", action="store_true")
args = parser.parse_args()

model = TransNetV2(device='auto')
model.eval()

input_dir = "./autodl-tmp/videos/"
base_output_dir = "./autodl-tmp/scenes/"
os.makedirs(base_output_dir, exist_ok=True)

video_files = [f for f in sorted(os.listdir(input_dir)) if f.endswith((".mp4", ".mkv", ".avi", ".mov"))]
total_videos = len(video_files)

for video_idx, video_name in enumerate(video_files, 1):
    # Create separate output folder for each video
    video_basename = os.path.splitext(video_name)[0]
    video_output_dir = os.path.join(base_output_dir, video_basename)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Skip already processed videos
    scenes_length_path = os.path.join(video_output_dir, "scenes_length.json")
    if os.path.exists(scenes_length_path):
        print(f"[{video_idx}/{total_videos}] [INFO] Skipping {video_name} (already processed)")
        continue
    
    video_path = os.path.join(input_dir, video_name)
    
    print(f"[{video_idx}/{total_videos}] [INFO] Processing {video_name}...")
    
    with torch.inference_mode():
        results = model.analyze_video(video_path)
        scenes = results['scenes']
        fps = results['fps']
    
    scenes_length = {}
    total_scenes = len(scenes)
    
    for i, scene in enumerate(scenes):
        start_frame = scene['start_frame']
        end_frame = scene['end_frame']
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps
        scene_filename = f"{video_basename}_scene{i+1:03d}.mp4"
        output_file = os.path.join(video_output_dir, scene_filename)
        
        (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .output(output_file, map_metadata="-1")
            .run(overwrite_output=True, quiet=True)
        )
        
        scenes_length[scene_filename] = round(duration, 3)
        
        if (i + 1) % 10 == 0 or i + 1 == total_scenes:
            print(f"[{video_idx}/{total_videos}] [INFO] Progress: {i+1}/{total_scenes} scenes")
    
    with open(scenes_length_path, "w", encoding="utf-8") as f:
        json.dump(scenes_length, f, ensure_ascii=False, indent=4)
    
    print(f"[{video_idx}/{total_videos}] [DONE] Saved {total_scenes} scenes to {video_output_dir}")
    
    if args.delete_source:
        os.remove(video_path)
        print(f"[{video_idx}/{total_videos}] [INFO] Deleted source file: {video_name}")
    
print("[DONE] All videos have been processed")
