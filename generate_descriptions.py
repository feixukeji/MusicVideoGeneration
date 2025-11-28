from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import json
import gc

# 配置参数
SCENES_DIR = "./autodl-tmp/scenes/"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MIN_VIDEO_LENGTH = 1
MAX_VIDEO_LENGTH = 20
MAX_PIXELS = 480 * 360
FPS = 8.0
TEXT_PROMPT = "简要描述这个视频"


def load_model_and_processor():
    """加载模型和处理器"""
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def load_descriptions(descriptions_path, video_folder):
    """加载已有的描述文件"""
    if os.path.exists(descriptions_path):
        with open(descriptions_path, "r", encoding="utf-8") as f:
            descriptions = json.load(f)
        print(f"Loaded existing descriptions.json for {video_folder} with {len(descriptions)} entries")
        return descriptions
    return {}


def save_descriptions(descriptions_path, descriptions):
    """保存描述到文件"""
    with open(descriptions_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=4)


def generate_description(model, processor, video_path):
    """为单个视频生成描述"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": MAX_PIXELS,
                    "fps": FPS,
                },
                {"type": "text", "text": TEXT_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True
    )

    if video_inputs is not None:
        videos, video_metadatas = zip(*video_inputs)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        videos, video_metadatas = None, None
    print(video_metadatas)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=videos,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # 清理显存
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()
    gc.collect()
    
    return output_text[0]


def process_video_folder(model, processor, folder_path, video_folder):
    """处理单个视频文件夹"""
    scenes_length_path = os.path.join(folder_path, "scenes_length.json")
    descriptions_path = os.path.join(folder_path, "descriptions.json")
    
    # 检查 scenes_length.json 是否存在
    if not os.path.exists(scenes_length_path):
        print(f"Warning: {scenes_length_path} not found, skipping folder {video_folder}")
        return
    
    descriptions = load_descriptions(descriptions_path, video_folder)
    
    with open(scenes_length_path, "r", encoding="utf-8") as f:
        scenes_lengths = json.load(f)
    
    for filename, video_length in scenes_lengths.items():
        # 跳过已经处理过的片段
        if filename in descriptions:
            print(f"Skipping {filename} as it has already been processed.")
            continue
        
        video_path = os.path.join(folder_path, filename)
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found, skipping.")
            continue
        
        if video_length < MIN_VIDEO_LENGTH or video_length > MAX_VIDEO_LENGTH:
            print(f"Skip {filename} because its length is {video_length} seconds.")
            continue

        print(f"Processing {filename}...")
        
        try:
            description = generate_description(model, processor, video_path)
            descriptions[filename] = description
            save_descriptions(descriptions_path, descriptions)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # 清理显存后继续处理下一个
            torch.cuda.empty_cache()
            gc.collect()
            continue
    
    print(f"All scenes for {video_folder} have been processed.")


def main():
    """主函数"""
    print("Loading model and processor...")
    model, processor = load_model_and_processor()
    
    # 获取所有视频文件夹
    video_folders = [
        f for f in os.listdir(SCENES_DIR) 
        if os.path.isdir(os.path.join(SCENES_DIR, f))
    ]
    
    if not video_folders:
        print(f"No video folders found in {SCENES_DIR}")
        return
    
    print(f"Found {len(video_folders)} video folder(s) to process.")
    
    for video_folder in video_folders:
        folder_path = os.path.join(SCENES_DIR, video_folder)
        process_video_folder(model, processor, folder_path, video_folder)
    
    print("All video folders have been processed.")


if __name__ == "__main__":
    main()
