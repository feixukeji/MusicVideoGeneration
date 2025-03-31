from transnetv2 import TransNetV2
import ffmpeg
import os
import json
import subprocess

model = TransNetV2(model_dir="./TransNetV2/inference/transnetv2-weights/")

input_dir = "./autodl-tmp/videos/"
base_output_dir = "./autodl-tmp/scenes/"
os.makedirs(base_output_dir, exist_ok=True)

video_files = sorted(os.listdir(input_dir))

def get_video_duration(video_path):
    """获取视频时长"""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return float(result.stdout)

for video_name in video_files:
    if not video_name.endswith((".mp4", ".mkv", ".avi", ".mov")):
        continue
    
    # 为每个视频创建单独的输出文件夹
    video_basename = os.path.splitext(video_name)[0]
    video_output_dir = os.path.join(base_output_dir, video_basename)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 检查是否已处理过该视频（断点续传）
    scenes_length_path = os.path.join(video_output_dir, "scenes_length.json")
    if os.path.exists(scenes_length_path):
        print(f"Skipping {video_name} as it has already been processed.")
        continue
    
    video_path = os.path.join(input_dir, video_name)
    
    print(f"Processing {video_name}...")
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
    scenes = model.predictions_to_scenes(single_frame_predictions)
    
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    fps = eval(video_stream['r_frame_rate'])
    
    scenes_length = {}
    
    for i, (start, end) in enumerate(scenes):
        start_time = start / fps
        duration = (end - start) / fps
        scene_filename = f"{video_basename}_scene{i+1:03d}.mp4"
        output_file = os.path.join(video_output_dir, scene_filename)
        
        ffmpeg.input(video_path, ss=start_time, t=duration).output(output_file, map_metadata="-1").run(overwrite_output=True)
        
        # 计算并保存场景时长
        scene_duration = get_video_duration(output_file)
        scenes_length[scene_filename] = scene_duration
    
    # 保存scenes_length.json
    with open(scenes_length_path, "w", encoding="utf-8") as f:
        json.dump(scenes_length, f, ensure_ascii=False, indent=4)
    
    print(f"Scenes length for {video_name} saved to {scenes_length_path}")
    
print("All videos have been processed.")