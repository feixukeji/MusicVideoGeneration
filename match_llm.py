import json
from openai import OpenAI
import re

client = OpenAI(api_key="api_key",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
model_name = "qwen-max"

lrc_path = "./autodl-tmp/lyrics.lrc"
scenes_length_path = "./autodl-tmp/scenes_length.json"
descriptions_path = "./autodl-tmp/descriptions.json"
best_matches_path = "./autodl-tmp/best_matches.json"

# 读取歌词
with open(lrc_path, "r", encoding="utf-8") as f:
    lyrics = f.readlines()


def parse_lyrics(lyrics):
    parsed_lyrics = []
    for line in lyrics:
        match = re.match(r'\[(\d+):(\d+\.\d+)\](.*)', line)
        if match:
            timestamp = int(match.group(1)) * 60 + float(match.group(2))
            content = match.group(3)
            parsed_lyrics.append((timestamp, content.strip()))
    return parsed_lyrics


parsed_lyrics = parse_lyrics(lyrics)

# 计算歌词时长
lyric_durations = []
total_duration = 0
for i in range(len(parsed_lyrics) - 1):
    duration = parsed_lyrics[i+1][0] - parsed_lyrics[i][0]
    lyric_durations.append(duration)
    total_duration += duration

# 计算平均时长
average_duration = total_duration / (len(parsed_lyrics) - 1)

# 最后一首歌词的时长
last_lyric_duration = average_duration * 1.2
lyric_durations.append(last_lyric_duration)

# Load scenes lengths
with open(scenes_length_path, "r", encoding="utf-8") as f:
    scenes_lengths = json.load(f)

# 读取视频描述
with open(descriptions_path, "r", encoding="utf-8") as f:
    descriptions = json.load(f)

# 去除视频描述中的回车
for key, value in descriptions.items():
    descriptions[key] = value.replace('\n', '')


def find_best_match(lyric, descriptions):
    system_prompt = '你将收到若干个视频片段的描述，需要为歌词"'+lyric+'''"搭配一个视频片段，使用如下 JSON 格式输出你的回复：
{
    "Think": "深入、详细的思考、分析、推理过程",
    "Answer": 最契合的视频片段的编号（一个数即可，不含解释）
}'''
    prompt = ""
    for i, desc in enumerate(descriptions):
        prompt += f"{i+1}. {desc[1]}\n"
    print(system_prompt)
    print(prompt)

    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                stream=False
            )
            response_text = response.choices[0].message.content
            print(response_text)
            response_id = json.loads(response_text)["Answer"]
            best_match = descriptions[response_id - 1]
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
    print(best_match)
    return best_match


best_matches = {}

# 读取最佳匹配结果
try:
    with open(best_matches_path, "r", encoding="utf-8") as f:
        best_matches = json.load(f)
except FileNotFoundError:
    best_matches = {}

# 遍历每句歌词
for i, (timestamp, lyric) in enumerate(parsed_lyrics):
    # 跳过已处理的歌词
    if str(timestamp) in best_matches:
        continue

    print(f"Processing lyric {i+1}/{len(parsed_lyrics)}: {lyric}")
    # 获取当前歌词的持续时间
    lyric_duration = lyric_durations[i]

    # 筛选出时长与歌词时长误差在以内的视频片段
    candidate_descriptions = []
    for desc in descriptions.items():
        scene_length = scenes_lengths[desc[0]]

        # 计算时长误差
        duration_error = abs(scene_length - lyric_duration) / lyric_duration

        # 如果误差在以内，则将该视频片段添加到候选列表中
        if duration_error <= 0.2:
            candidate_descriptions.append(desc)

    # 如果没有符合条件的视频片段，则跳过
    if not candidate_descriptions:
        print(f"No suitable scenes found for lyric: {lyric}")
        best_matches[str(timestamp)] = "None"
        continue

    # 从候选列表中选择最佳匹配
    print(len(candidate_descriptions))
    if len(candidate_descriptions) > 50:
        group_size = int(len(candidate_descriptions) ** 0.5)
        best_match_list = []
        candidate_descriptions_grouped = [candidate_descriptions[i:i+group_size]
                                          for i in range(0, len(candidate_descriptions), group_size)]
        for group in candidate_descriptions_grouped:
            best_match = find_best_match(lyric, group)
            best_match_list.append(best_match)
        print(best_match_list)
        best_match = find_best_match(lyric, best_match_list)
        print(best_match)
        best_matches[str(timestamp)] = best_match[0]
    else:
        best_match = find_best_match(lyric, candidate_descriptions)
        print(best_match)
        best_matches[str(timestamp)] = best_match[0]

    # 保存最佳匹配结果
    with open(best_matches_path, "w", encoding="utf-8") as f:
        json.dump(best_matches, f, ensure_ascii=False, indent=4)

print("Best matches saved to best_matches.json")
