# Open-Domain QA

# ディレクトリ構造

```yaml
- datasets.yml:                        データセット定義

# 前処理
- prepro/:
  - __init__.py:
  - convert_dataset.py:                データ形式変換
  - JaQuAD/:
    - load_jaquad.py:                  JaQuAD ロード
  - JaqketAIO/:
    - download_data.sh:                Jaqket ダウンロード
    - load_jaqketaio2.py:              Jaqket ロード 

# TF-IDF ほか
- vectorizers/:
  - aio2_tfidf_baseline/:              TF-IDF ベースライン
  - densify_sparse_representations.py: DSRs

# Retriever
- retrievers/:
  - AIO2_DPR_baseline/:                DPR ベースライン (AIO2)
  - aio2-soseki-baseline/:             BPR ベースライン (AIO2)

# Indexing
- indexers: 
  - README.md: FAISS について

# 抽出型 Reader
- readers/:

# 生成型 Reader
- generators/:
  - fusion_in_decoder/: FiD

# Re-Ranker
- rerankers/:
```

# References
- https://github.com/terrier-org/pyterrier

# Tips

## Git CLI
- [Git submodule の基礎 (Qiita)](https://qiita.com/sotarok/items/0d525e568a6088f6f6bb)
