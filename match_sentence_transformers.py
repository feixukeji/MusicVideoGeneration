import json
import re
import os
import torch
from sentence_transformers import SentenceTransformer

LRC_PATH = "./autodl-tmp/lyrics.lrc"
SCENES_DIR = "./autodl-tmp/scenes/"
BEST_MATCHES_PATH = "./autodl-tmp/best_matches.txt"
TOP_K = 3  # Save top K best matches
MODEL_NAME = 'Qwen/Qwen3-Embedding-8B'
TASK_INSTRUCTION = 'Given a poetic lyric, retrieve a video description that visually represents the metaphor or scene implied.'


def parse_lyrics(lyrics_lines: list[str]) -> tuple[list[float], list[str]]:
    """Parse lyrics file and extract timestamps and content."""
    timestamps = []
    parsed_lyrics = []
    pattern = re.compile(r'\[(\d+):(\d+\.\d+)\](.*)')
    
    for line in lyrics_lines:
        match = pattern.match(line)
        if match:
            timestamp = int(match.group(1)) * 60 + float(match.group(2))
            content = match.group(3).strip()
            if content:  # Skip empty lyrics
                timestamps.append(timestamp)
                parsed_lyrics.append(content)
    return timestamps, parsed_lyrics


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Generate query format with instruction."""
    return f'Instruct: {task_description}\nQuery: {query}'


def load_descriptions(scenes_dir: str) -> dict[str, str]:
    """Load all scene descriptions and remove newlines."""
    descriptions = {}
    
    if not os.path.exists(scenes_dir):
        raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")
    
    video_folders = [f for f in os.listdir(scenes_dir) 
                     if os.path.isdir(os.path.join(scenes_dir, f))]
    
    print(f"[INFO] Found {len(video_folders)} video folder(s) to process")
    
    for video_folder in video_folders:
        descriptions_path = os.path.join(scenes_dir, video_folder, "descriptions.json")
        
        if not os.path.exists(descriptions_path):
            print(f"[WARNING] Description file {descriptions_path} not found, skipping")
            continue
        
        try:
            with open(descriptions_path, "r", encoding="utf-8") as f:
                folder_descriptions = json.load(f)
            # Remove newlines from descriptions
            for key, value in folder_descriptions.items():
                descriptions[key] = value.replace('\n', '')
        except json.JSONDecodeError as e:
            print(f"[WARNING] Failed to parse {descriptions_path}: {e}")
            continue
    
    return descriptions


if __name__ == "__main__":
    # Read and parse lyrics
    with open(LRC_PATH, "r", encoding="utf-8") as f:
        lyrics = f.readlines()
    
    timestamps, parsed_lyrics = parse_lyrics(lyrics)
    
    # Load scene descriptions
    descriptions = load_descriptions(SCENES_DIR)
    
    if not descriptions:
        raise ValueError("No scene descriptions loaded")
    
    # Prepare queries and documents
    queries = [get_detailed_instruct(TASK_INSTRUCTION, lyric) for lyric in parsed_lyrics]
    documents = list(descriptions.values())
    scene_keys = list(descriptions.keys())
    
    print(f"[INFO] Processing {len(parsed_lyrics)} lyrics and {len(documents)} scene descriptions")
    
    # Load model
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    
    # Encode queries and documents with progress bar
    print("[INFO] Encoding queries...")
    query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
    print("[INFO] Encoding documents...")
    document_embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
    
    # Calculate similarity scores
    scores = model.similarity(query_embeddings, document_embeddings)
    
    # Get best matches
    top_scores, top_indices = torch.topk(scores, k=min(TOP_K, len(documents)), dim=1)
    
    # Save results
    with open(BEST_MATCHES_PATH, "w", encoding="utf-8") as f:
        for i, lyric in enumerate(parsed_lyrics):
            f.write(f"Lyric: {lyric}\n")
            for rank in range(top_indices.shape[1]):
                idx = top_indices[i, rank].item()
                score = top_scores[i, rank].item()
                f.write(f"  Match {rank + 1}: {scene_keys[idx]} (Score: {score:.4f})\n")
            f.write("\n")
    
    print(f"[DONE] Top {TOP_K} matches for each lyric saved to {BEST_MATCHES_PATH}")