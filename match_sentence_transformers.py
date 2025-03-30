import json
import re
import os
from sentence_transformers import SentenceTransformer

lrc_path = "./autodl-tmp/lyrics.lrc"
scenes_dir = "./autodl-tmp/scenes/"
best_matches_path = "./autodl-tmp/best_matches.json"

# 读取歌词
with open(lrc_path, "r", encoding="utf-8") as f:
    lyrics = f.readlines()


def parse_lyrics(lyrics):
    timestamps = []
    parsed_lyrics = []
    for line in lyrics:
        match = re.match(r'\[(\d+):(\d+\.\d+)\](.*)', line)
        if match:
            timestamp = int(match.group(1)) * 60 + float(match.group(2))
            content = match.group(3)
            timestamps.append(timestamp)
            parsed_lyrics.append(content.strip())
    return timestamps, parsed_lyrics


timestamps, parsed_lyrics = parse_lyrics(lyrics)

# 汇总所有子文件夹的descriptions.json
descriptions = {}
video_folders = [f for f in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, f))]

print(f"Found {len(video_folders)} video folders to process")

for video_folder in video_folders:
    folder_path = os.path.join(scenes_dir, video_folder)
    descriptions_path = os.path.join(folder_path, "descriptions.json")
    
    with open(descriptions_path, "r", encoding="utf-8") as f:
        folder_descriptions = json.load(f)
    
    descriptions.update(folder_descriptions)

# 去除视频描述中的回车
for key, value in descriptions.items():
    descriptions[key] = value.replace('\n', '')


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# Each query must come with a one-sentence instruction that describes the task
task = 'Given a lyric, retrieve the relevant passage that match the lyric'
queries = [get_detailed_instruct(task, lyric) for lyric in parsed_lyrics]
# No need to add instruction for retrieval documents
documents = list(descriptions.values())
input_texts = queries + documents

print(f"Processing {len(parsed_lyrics)} lyrics and {len(documents)} scene descriptions")

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

embeddings = model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)
scores = embeddings[:len(queries)] @ embeddings[len(queries):].T

best_matches = {}
for i, lyric in enumerate(parsed_lyrics):
    best_match_index = scores[i].argmax().item()
    best_matches[lyric] = list(descriptions.keys())[best_match_index]

# 保存最佳匹配结果
with open(best_matches_path, "w", encoding="utf-8") as f:
    json.dump(best_matches, f, ensure_ascii=False, indent=4)
print(f"Best matches saved to {best_matches_path}.")