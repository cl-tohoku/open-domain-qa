# Fusion-in-Decoder (FiD)

- This directory is forked by [facebookresearch/FiD](https://github.com/facebookresearch/FiD)
- Izacard+'21 - Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering (EACL) [[arXiv](https://arxiv.org/abs/2007.01282)]
  - Yamada+'21 - オープンドメイン質問応答技術の最新動向 (AI王2021) [[Speaker Deck](https://speakerdeck.com/ikuyamada/opundomeinzhi-wen-ying-da-ji-shu-falsezui-xin-dong-xiang?slide=39)]
    > <img src="https://i.gyazo.com/f8771cde1ad3322d59d31ea8c11c6f02.png" alt="fid" title="fusion-in-decoder">


## 設定

```bash
$ cd aobav2/aoba/dialogs/fid

# pyenv local しておく
$ conda create -n FiD python=3.8 -y
$ pyenv local ${env}/FiD
$ bash scripts/set_env.sh
```

## データセット

### 取得

1. `{ROOT_REPOSITORY}/datasets.yml` ファイルに以下を追記

```yml
DprRetrieved:
  path: JaqketAIO.load_jaqketaio2
  class: JaqketAIO
  data:
    train: {retrieved_train}
    dev: {retrieved_dev}
    test: {retrieved_test}
    unused: {retrieved_unused}
```

```bash
$ cd {ROOT_REPOSITORY}
$ python prepro/convert_dataset.py DprRetrieved fusion_in_decoder
```

### 形式
以下のインスタンスからなる JSON/JSONL ファイルを使用

```json
{
    "id": "(str) 質問ID",
    "question": "(str) 質問",
    "target": "(str) answers から一つ選択した答え。ない場合はランダムに選択される。",
    "answers": "(List[str]) 答えのリスト",
    "ctxs": [{
        "title": "(str) Wikipedia 記事タイトル",
        "text": "(str) Wikipedia 記事",
        "score": "(float) retriever の検索スコア (ない場合は 1/idx で置換される。generator では使用されない。)"
    }]
}
```


## 学習

```bash
$ bash scripts/train_generator.sh configs/train_generator_slud.yml
```

## 評価

```bash
$ bash scripts/train_generator.sh configs/train_generator_slud.yml
```
