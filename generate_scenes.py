import torch
from transnetv2_pytorch import TransNetV2
import ffmpeg
import os
import json

model = TransNetV2(device='auto')
model.eval()

input_dir = "./autodl-tmp/videos/"
base_output_dir = "./autodl-tmp/scenes/"
os.makedirs(base_output_dir, exist_ok=True)

video_files = [f for f in sorted(os.listdir(input_dir)) if f.endswith((".mp4", ".mkv", ".avi", ".mov"))]
total_videos = len(video_files)

for video_idx, video_name in enumerate(video_files, 1):
    # 为每个视频创建单独的输出文件夹
    video_basename = os.path.splitext(video_name)[0]
    video_output_dir = os.path.join(base_output_dir, video_basename)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 跳过已经处理过的视频
    scenes_length_path = os.path.join(video_output_dir, "scenes_length.json")
    if os.path.exists(scenes_length_path):
        print(f"[{video_idx}/{total_videos}] Skipping {video_name} (already processed)")
        continue
    
    video_path = os.path.join(input_dir, video_name)
    
    print(f"[{video_idx}/{total_videos}] Processing {video_name}...")
    
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
            print(f"  Scenes: {i+1}/{total_scenes}")
    
    with open(scenes_length_path, "w", encoding="utf-8") as f:
        json.dump(scenes_length, f, ensure_ascii=False, indent=4)
    
    print(f"  Saved {total_scenes} scenes to {video_output_dir}")
    
print("All videos have been processed.")