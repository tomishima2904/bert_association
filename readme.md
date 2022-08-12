# 使い方
1. パラメータを決める(configにデフォルトが設定してある)
2. bert_experiment.pyを実行する
```
python bert_experiment.py \
	--analysis_flag=True \
	--avg_flag=True \
	--brackets_flag=True \
	--dataset='extract_keywordslist' \
	--dict_mecab='ipadic' \
	--eval_opt='p' \
	--extract_noun_opt='mecab' \
	--framework_opt='tf' \
	--max_words=150 \
	--model_opt='cl-tohoku' \
	--num_stims=5 \
	--output_nayose_flag=True \
	--output_words_from_bert=True \
	--sep_flag='f' \
	--target_heads -1 \
	--target_layer=-1 \
    --reverse_flag='f' \
	--category_opt='cat'
```
        - flag系は`True`や`t`もしくは`False`や`f`で選択
        - `target_heads`等はリストが引数となる．`[2, 4, 5]`のように引数を渡したければ`--target_heads 2 4 5`とする
3. result内に結果が出力される


# requirements
```
transformers==3.1.0
tensorflow-gpu==2.3.0
pandas==1.1.1
spacy==3.2.1
torch
fugashi
mecab-python3
ipadic
#-U ginza ja-ginza
tqdm
numpy
matplotlib
scipy
sklearn
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
#### addparts
名詞のみで事前学習したBERT。
基本設定は東北大学BERTと同じだが、事前学習データセットとして使用したWikipediaのバージョンが東北大学BERTよりも1年くらい新しい。

#### cl-tohoku
東北大学BERT。
ただし、中身はvocab.txtだけ。
モデル本体はhugging faceのtransformersライブラリに内蔵されている。

#### dataset
実験で使用する単語や名寄せファイルが入ったディレクトリ。

#### gMLP
名詞のみで事前学習したgMLP。
モデル本体はofficialディレクトリの何処かにある。
最終結果しか出力しないので何も分析できない。
使わない方が無難。ごめんなさい。

#### official
gMLPのモデル本体が入ったディレクトリ。
nlp/gmlpはモデル本体、
modeling/models/gmlp_pretrainer.pyは学習用ファイル

#### result
結果を保存するディレクトリ。

#### 参考資料
喚語資料とか87題に選定した際のエクセルファイルとか。

#### bert_experiment.py
複数→1つの連想実験用のファイル。

#### models.py
モデルが記述されているファイル。

#### dict_maker.py
dataset内のファイルをdict型に変換するファイル。

#### utils_tools.py
データの前処理や連想文が記載されたファイル。

#### analysis.py および analysis2.py
分析を行うファイル。attentionの可視化等を行う。

#### file_handler.py
ファイルを読み込んだり、書き出したり、パスの名前を設定するファイル。

#### config.py
パラメータを設定するファイル。


# 出力ファイルについて
#### result.csv
sid(sentence id), stims(sitimulations; 刺激語), input_sentence, answer(正解の連想語), category, category_synonyms(カテゴリーの類似語), output_words(刺激語を除いたり名寄せしたバージョン),output_scores(刺激語を除いたり名寄せしたバージョン)

#### result_attentions_and_namas.csv
sid, stims, input_sentence, tokenized_sentence(トークナイズされた入力文), answer, category, category_synonyms, attn_weights_of_mask(MASKトークンをqueryとした時のattention weight), output_raw_words(後処理しないバージョン), output_raw_scores(後処理しないバージョン)

#### analysis.csv(一見謎のファイルだが、1つ→複数の名残なので気にしないで)
sid, stims, input_sentence, answer, category, category_synonyms, ranks, corr_word(正解の連想語), corr_score(正解語の出力スコア), err_words(不正解語), err_scores(不正解語の出力スコア), num_err_per_iv(よくわからない)

#### analysis_analysis.csv(csvファイルだが、中身が全然csvファイルじゃない。使っていない)
連想文番号
カテゴリ名
上位n語
正解率
上位n語に正解語が出力されれば1.0、されていないと0.0