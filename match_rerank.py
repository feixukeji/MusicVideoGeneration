import dashscope
import json

descriptions_path = "./autodl-tmp/descriptions.json"

dashscope.api_key  = "api_key"

with open(descriptions_path, "r", encoding="utf-8") as f:
    descriptions = json.load(f)

# Process in batches of 500
batch_size = 500
best_match = None
best_score = -1
descriptions_items = list(descriptions.items())

for i in range(0, len(descriptions_items), batch_size):
    batch = descriptions_items[i:i + batch_size]
    descriptions_list = [item[1] for item in batch]
    
    resp = dashscope.TextReRank.call(
        model=dashscope.TextReRank.Models.gte_rerank, 
        query="歌词",
        documents=descriptions_list,
        top_n=1
    )
    
    batch_best_score = resp.output.results[0].relevance_score
    batch_best_index = resp.output.results[0].index
    print(f"Best match in batch: {batch[batch_best_index]}")
    print(f"Best score in batch: {batch_best_score}")
    
    if batch_best_score > best_score:
        best_score = batch_best_score
        best_match = batch[batch_best_index]

print(f"Best match overall: {best_match}")
print(f"Best score: {best_score}")