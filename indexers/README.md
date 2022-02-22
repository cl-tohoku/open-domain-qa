# Indexers

## faiss

ベクトルを用いた類似度検索で有効な近傍探索ライブラリ

```bash
$ pip install -U faiss-gpu
```

### References

- https://github.com/facebookresearch/faiss/wiki/
- https://github.com/facebookresearch/faiss/wiki/Faiss-indexes


### How to use

```python
from typing import Any, List

from cytoolz import curry
import faiss
import numpy as np
from transformers import BertJapaneseTokenizer, BertModel

# get vectors
model_name_or_path = "cl-tohoku/bert-base-japanese-v2"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
model = BertModel.from_pretrained(model_name_or_path)

query = ["インフルエンザの予防接種受けましたか？", "バスケは好きですか"]  # クエリ
corpus = ["今日はいい天気だね", "私はサッカーが好きです", "コロナのワクチン打ってきました"]  # 検索対象

top_k = 3  # 検索数


@curry
def encode(batch_text: List[str], model: Any, tokenizer: Any) -> np.array:
    if isinstance(batch_text, str):
        return encode([batch_text])
    
    inputs = tokenizer.batch_encode_plus(
        batch_text, 
        pad_to_max_length=True, 
        add_special_tokens=True, 
        return_tensors="pt"
    )

    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"],
    )

    return outputs["pooler_output"].detach().numpy()


encode_fn = encode(model=model, tokenizer=tokenizer)

query_vector: np.array = encode_fn(query)    # size: [2, 768]
corpus_vector: np.array = encode_fn(corpus)  # size: [3, 768]

_cosine_similarity = True
if _cosine_similarity == True:
    faiss.normalize_L2(query_vector)
    faiss.normalize_L2(corpus_vector)

hidden_size = model.config.hidden_size  # 768
indexer = faiss.IndexFlatIP(hidden_size)

indexer.add(corpus_vector)                       # add to indexer
results = indexer.search(query_vector, k=top_k)  # search

for qid, (distances, indices) in enumerate(zip(*results)):
    # distances: 類似度スコア (IndexFlatIP では内積値を計算)
    # indices: indexer に登録されたコーパス番号
    print(f"query ... {query[qid]}")
    for top_k, (d, i) in enumerate(zip(distances, indices)):
        print(f"  [{top_k}] (score: {d:.4f}) {corpus[i]}")

"""
query ... インフルエンザの予防接種受けましたか？
  [0] (score: 135.4354) 私はサッカーが好きです
  [1] (score: 107.6180) コロナのワクチン打ってきました
  [2] (score: 67.6759) 今日はいい天気だね
query ... バスケは好きですか
  [0] (score: 143.9923) 私はサッカーが好きです
  [1] (score: 98.0991) コロナのワクチン打ってきました
  [2] (score: 61.7178) 今日はいい天気だね
"""
```
