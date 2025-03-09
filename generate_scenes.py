from transnetv2 import TransNetV2
import ffmpeg
import os

model = TransNetV2(model_dir="./TransNetV2/inference/transnetv2-weights/")

input_dir = "./autodl-tmp/videos/"
output_dir = "./autodl-tmp/scenes/"
os.makedirs(output_dir, exist_ok=True)

video_files = sorted(os.listdir(input_dir))

for video_name in video_files:
    if not video_name.endswith((".mp4", ".mkv", ".avi", ".mov")):
        continue
    video_path = os.path.join(input_dir, video_name)

    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
    scenes = model.predictions_to_scenes(single_frame_predictions)

    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    fps = eval(video_stream['r_frame_rate'])
    print(f"Video frame rate: {fps}")

    for i, (start, end) in enumerate(scenes):
        start_time = start / fps
        duration = (end - start) / fps
        output_file = os.path.join(output_dir, os.path.splitext(video_name)[0] + f"_scene{i+1:03d}.mp4")

        ffmpeg.input(video_path, ss=start_time, t=duration).output(output_file, map_metadata="-1").run()

        print(f"Saved scene {i+1} from frame {start} to {end} as {output_file}")