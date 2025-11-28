import json
import re
import os
import torch
from sentence_transformers import SentenceTransformer

# ============== 配置参数 ==============
LRC_PATH = "./autodl-tmp/lyrics.lrc"
SCENES_DIR = "./autodl-tmp/scenes/"
BEST_MATCHES_PATH = "./autodl-tmp/best_matches.txt"
TOP_K = 3  # 保存前K个最佳匹配
MODEL_NAME = 'Qwen/Qwen3-Embedding-8B'
TASK_INSTRUCTION = 'Given a poetic lyric, retrieve a video description that visually represents the metaphor or scene implied.'


# ============== 函数定义 ==============
def parse_lyrics(lyrics_lines: list[str]) -> tuple[list[float], list[str]]:
    """解析歌词文件，提取时间戳和歌词内容"""
    timestamps = []
    parsed_lyrics = []
    pattern = re.compile(r'\[(\d+):(\d+\.\d+)\](.*)')
    
    for line in lyrics_lines:
        match = pattern.match(line)
        if match:
            timestamp = int(match.group(1)) * 60 + float(match.group(2))
            content = match.group(3).strip()
            if content:  # 跳过空歌词
                timestamps.append(timestamp)
                parsed_lyrics.append(content)
    return timestamps, parsed_lyrics


def get_detailed_instruct(task_description: str, query: str) -> str:
    """生成带指令的查询格式"""
    return f'Instruct: {task_description}\nQuery: {query}'


def load_descriptions(scenes_dir: str) -> dict[str, str]:
    """加载所有场景描述，去除换行符"""
    descriptions = {}
    
    if not os.path.exists(scenes_dir):
        raise FileNotFoundError(f"场景目录不存在: {scenes_dir}")
    
    video_folders = [f for f in os.listdir(scenes_dir) 
                     if os.path.isdir(os.path.join(scenes_dir, f))]
    
    print(f"Found {len(video_folders)} video folders to process")
    
    for video_folder in video_folders:
        descriptions_path = os.path.join(scenes_dir, video_folder, "descriptions.json")
        
        if not os.path.exists(descriptions_path):
            print(f"警告: 未找到描述文件 {descriptions_path}，跳过")
            continue
        
        try:
            with open(descriptions_path, "r", encoding="utf-8") as f:
                folder_descriptions = json.load(f)
            # 去除描述中的换行符
            for key, value in folder_descriptions.items():
                descriptions[key] = value.replace('\n', '')
        except json.JSONDecodeError as e:
            print(f"警告: 解析 {descriptions_path} 失败: {e}")
            continue
    
    return descriptions


# ============== 主程序 ==============
if __name__ == "__main__":
    # 读取并解析歌词
    with open(LRC_PATH, "r", encoding="utf-8") as f:
        lyrics = f.readlines()
    
    timestamps, parsed_lyrics = parse_lyrics(lyrics)
    
    # 加载场景描述
    descriptions = load_descriptions(SCENES_DIR)
    
    if not descriptions:
        raise ValueError("未加载到任何场景描述")
    
    # 准备查询和文档
    queries = [get_detailed_instruct(TASK_INSTRUCTION, lyric) for lyric in parsed_lyrics]
    documents = list(descriptions.values())
    scene_keys = list(descriptions.keys())
    
    print(f"Processing {len(parsed_lyrics)} lyrics and {len(documents)} scene descriptions")
    
    # 加载模型
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    
    # 编码查询和文档（添加进度条）
    print("Encoding queries...")
    query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
    print("Encoding documents...")
    document_embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
    
    # 计算相似度分数
    scores = model.similarity(query_embeddings, document_embeddings)
    
    # 获取最佳匹配
    top_scores, top_indices = torch.topk(scores, k=min(TOP_K, len(documents)), dim=1)
    
    # 保存结果
    with open(BEST_MATCHES_PATH, "w", encoding="utf-8") as f:
        for i, lyric in enumerate(parsed_lyrics):
            f.write(f"歌词: {lyric}\n")
            for rank in range(top_indices.shape[1]):
                idx = top_indices[i, rank].item()
                score = top_scores[i, rank].item()
                f.write(f"  匹配{rank + 1}: {scene_keys[idx]} (分数: {score:.4f})\n")
            f.write("\n")
    
    print(f"Top {TOP_K} matches for each lyric saved to {BEST_MATCHES_PATH}.")