from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import json

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

scenes_dir = "./autodl-tmp/scenes/"
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# 获取所有视频文件夹
video_folders = [f for f in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, f))]

for video_folder in video_folders:
    folder_path = os.path.join(scenes_dir, video_folder)
    
    scenes_length_path = os.path.join(folder_path, "scenes_length.json")
    descriptions_path = os.path.join(folder_path, "descriptions.json")
    
    # 加载现有的descriptions.json（如果存在）
    descriptions = {}
    if os.path.exists(descriptions_path):
        with open(descriptions_path, "r", encoding="utf-8") as f:
            descriptions = json.load(f)
        print(f"Loaded existing descriptions.json for {video_folder} with {len(descriptions)} entries")
    
    # 加载该文件夹的scenes_length.json
    with open(scenes_length_path, "r", encoding="utf-8") as f:
        scenes_lengths = json.load(f)
    
    for filename, video_length in scenes_lengths.items():
        # 断点续传：跳过已经处理过的片段
        if filename in descriptions:
            print(f"Skipping {filename} as it has already been processed.")
            continue
            
        video_path = os.path.join(folder_path, filename)
        
        if video_length < 1 or video_length > 20:
            print(f"Skip {filename} because its length is {video_length} seconds.")
            continue

        print(f"Processing {filename}...")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 480 * 360,
                        # "fps": 8.0,
                    },
                    {"type": "text", "text": "描述这个视频"},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        descriptions[filename] = output_text[0]
        
        # 每处理完一个片段就保存一次descriptions.json
        with open(descriptions_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=4)
    
    print(f"All scenes for {video_folder} have been processed.")

print("All video folders have been processed.")
