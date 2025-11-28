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

# Read lyrics
with open(lrc_path, "r", encoding="utf-8") as f:
    lyrics = f.readlines()


def parse_lyrics(lyrics):
    """Parse lyrics and extract timestamps with content."""
    parsed_lyrics = []
    for line in lyrics:
        match = re.match(r'\[(\d+):(\d+\.\d+)\](.*)', line)
        if match:
            timestamp = int(match.group(1)) * 60 + float(match.group(2))
            content = match.group(3)
            parsed_lyrics.append((timestamp, content.strip()))
    return parsed_lyrics


parsed_lyrics = parse_lyrics(lyrics)

# Calculate lyrics duration
lyric_durations = []
total_duration = 0
for i in range(len(parsed_lyrics) - 1):
    duration = parsed_lyrics[i+1][0] - parsed_lyrics[i][0]
    lyric_durations.append(duration)
    total_duration += duration

# Calculate average duration
average_duration = total_duration / (len(parsed_lyrics) - 1)

# Duration for the last lyric line
last_lyric_duration = average_duration * 1.2
lyric_durations.append(last_lyric_duration)

# Load scenes lengths
with open(scenes_length_path, "r", encoding="utf-8") as f:
    scenes_lengths = json.load(f)

# Read video descriptions
with open(descriptions_path, "r", encoding="utf-8") as f:
    descriptions = json.load(f)

# Remove newlines from video descriptions
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
            response_id = json.loads(response_text)["Answer"]
            best_match = descriptions[response_id - 1]
            break
        except Exception as e:
            print(f"[ERROR] {e}, retrying...")
    return best_match


best_matches = {}

# Load existing best matches
try:
    with open(best_matches_path, "r", encoding="utf-8") as f:
        best_matches = json.load(f)
except FileNotFoundError:
    best_matches = {}

# Iterate through each lyric line
for i, (timestamp, lyric) in enumerate(parsed_lyrics):
    # Skip already processed lyrics
    if str(timestamp) in best_matches:
        continue

    print(f"[{i+1}/{len(parsed_lyrics)}] [INFO] Processing lyric: {lyric}")
    # Get duration of current lyric
    lyric_duration = lyric_durations[i]

    # Filter video clips with duration error within threshold
    candidate_descriptions = []
    for desc in descriptions.items():
        scene_length = scenes_lengths[desc[0]]

        # Calculate duration error
        duration_error = abs(scene_length - lyric_duration) / lyric_duration

        # Add video clip to candidates if error is within threshold
        if duration_error <= 0.2:
            candidate_descriptions.append(desc)

    # Skip if no suitable video clips found
    if not candidate_descriptions:
        print(f"[WARNING] No suitable scenes found for lyric: {lyric}")
        best_matches[str(timestamp)] = "None"
        continue

    # Select best match from candidates
    print(f"[INFO] Found {len(candidate_descriptions)} candidate(s)")
    if len(candidate_descriptions) > 50:
        group_size = int(len(candidate_descriptions) ** 0.5)
        best_match_list = []
        candidate_descriptions_grouped = [candidate_descriptions[i:i+group_size]
                                          for i in range(0, len(candidate_descriptions), group_size)]
        for group in candidate_descriptions_grouped:
            best_match = find_best_match(lyric, group)
            best_match_list.append(best_match)
        best_match = find_best_match(lyric, best_match_list)
        best_matches[str(timestamp)] = best_match[0]
    else:
        best_match = find_best_match(lyric, candidate_descriptions)
        best_matches[str(timestamp)] = best_match[0]
    
    print(f"[{i+1}/{len(parsed_lyrics)}] [INFO] Best match: {best_match[0]}")

    # Save best matches
    with open(best_matches_path, "w", encoding="utf-8") as f:
        json.dump(best_matches, f, ensure_ascii=False, indent=4)

print("[DONE] Best matches saved to best_matches.json")
