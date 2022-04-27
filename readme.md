# 使い方
1. パラメータを決める(現在はbert_experiment.pyに直打ちしている。configファイルの使用を推奨)
2. bert_experiment.pyを実行する
```
python bert_experiment.py \
	--output_words_from_bert=True \
	--analysis_flag=True \
	--framework_opt='tf' \
	--model_opt='cl-tohoku' \
	--brackets_flag=True \
	--output_nayose_flag=True \
	--extract_noun_opt='mecab' \
	--multi_stimulations_flag=True \
	--category_flag=True \
	--num_stimulations=5 \
	--eval_opt='p' \
	--dataset='喚語資料_除去2' \
	--max_words=150 \
	--target_layer=-1 
```               
3. result内に結果が出力される


# 必要なライブラリ(requirement.txtを作るのがめんどくさかった...)
```
transformers==3.1.0
tensorflow-gpu==2.3.0
pandas==1.1.1
spacy==3.2.1
torch
mecab
fugashi
```


# mecabについて
東北大学BERTで使う形態素解析器はmecabだが、
使用する辞書に注意する必要がある。
Wikipediaコーパスでテキストをセンテンスに分解する際はmecab-ipadic-neologd(v0.0.7)を使用したが、
Tokenizerではunidic-liteを使用するらしい。(fugashiライブラリの中にunidic-liteが入っている。)

bert_experiment.pyではTokenizer以外にも名詞を判定するためにmecabを使用しているが、
相馬はipadicを使用していた...
(本当はunidicがいいけど、名詞を判定するだけならあまり変わらないので...)


# ディレクトリ/ファイルについて
##### addparts
名詞のみで事前学習したBERT。
基本設定は東北大学BERTと同じだが、事前学習データセットとして使用したWikipediaのバージョンが東北大学BERTよりも1年くらい新しい。

##### cl-tohoku
東北大学BERT。
ただし、中身はvocab.txtだけ。
モデル本体はhugging faceのtransformersライブラリに内蔵されている。

##### dataset
実験で使用する単語や名寄せファイルが入ったディレクトリ。

##### gMLP
名詞のみで事前学習したgMLP。
モデル本体はofficialディレクトリの何処かにある。
最終結果しか出力しないので何も分析できない。
使わない方が無難。ごめんなさい。

##### official
gMLPのモデル本体が入ったディレクトリ。
nlp/gmlpはモデル本体、
modeling/models/gmlp_pretrainer.pyは学習用ファイル

##### result
結果を保存するディレクトリ。

##### 参考資料
喚語資料とか87題に選定した際のエクセルファイルとか。

##### bert_experiment.py
複数→1つの連想実験用のファイル。

##### extract_hukusuu.py
dataset内のファイルをdict型に変換するファイル。

##### utils_tools.py
データの前処理や連想文が記載されたファイル。


# 出力ファイルについて
##### result.csv
刺激語, 連想文番号(初期設定では0のみ), 連想文(トークナイズ前), 正解の連想語, 出力単語(刺激語を除いたり名寄せしたバージョン), 出力スコア(刺激語を除いたり名寄せしたバージョン)

##### result_attentions_and_namas.csv
刺激語, 連想文番号(初期設定では0のみ), 連想文(トークナイズ前), 正解の連想語, Attention分析結果(初期設定ではNone), 出力単語(後処理しないバージョン), 出力スコア(後処理しないバージョン)

##### analysis.csv(一見謎のファイルだが、1つ→複数の名残なので気にしないで)
通し番号, 刺激語, 連想文番号(初期設定では0のみ), 連想文(トークナイズ前), 正解の連想語, 正解の連想語が出力された順位, 正解語, 正解語の出力スコア, 不正解語, 不正解の出力スコア

##### analysis_analysis.csv(csvファイルだが、中身が全然csvファイルじゃない)
連想文番号
カテゴリ名
上位n語
正解率
上位n語に正解語が出力されれば1.0、されていないと0.0