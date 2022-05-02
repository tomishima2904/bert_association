# 必要なライブラリ。 pipやcondaでインストールしてください。
from transformers import BertJapaneseTokenizer, TFBertForMaskedLM, BertConfig
import tensorflow as tf
import spacy
import csv
import ast
import os
import pandas as pd
import fugashi
import argparse
import datetime
import io,sys
from distutils.util import strtobool
from analysis import Analyzer
from file_handler import *

# 自作ファイルからのインポート
from extract_hukusuu import SearchFromHukusuuSigeki
import utils_tools
from models import Model

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# B4の頃から違法増築を繰り返した結果、一つのクラスに色々詰め込むゴミクラスが完成した...
class BertAssociation():
    # kwargsとか使った方がいい...
    def __init__(self, args):

        self.args = args

        self.model_opt = args.model_opt  # str. モデルを定める.
        self.framework_opt = args.framework_opt  # str. 使用するフレームワークを定める(学習しないのであまり区別する意味はないが...)

        # 問い語(=限定語)を使用するかどうか。
        # Trueの場合は「都道府県は～」「色は～」という連想文になる。
        # Falseの場合は「言葉は～」という連想文になる。
        # utils_tools.pyに連想文を書いてある。
        self.category_flag = args.category_flag

        # flagと書いてあるけどstr型その3。名詞を抽出する際に使用するライブラリ？を定める
        # mecab...MeCabを使用する
        # ginza...Ginzaを使用する
        self.extract_noun_opt = args.extract_noun_opt

        # 名詞判定で使用するmecabの設定(環境に合わせて設定してね。これは相馬のローカル環境の例)
        mecab_option = f"-d /usr/local/lib/python3.6/site-packages/{self.args.dict_mecab}/dicdir -r /usr/local/lib/python3.6/site-packages/{self.args.dict_mecab}/dicdir/mecabrc"

        if self.framework_opt == "tf":
            print(tf.__version__)
            model_object = Model(mecab_option, args)
            self.tokenizer = model_object.tokenizer
            self.model = model_object.model
            self.mecab = model_object.mecab
            
        # spacy の ginza モデルをインストール
        self.ginza = spacy.load('ja_ginza')

        if args.multi_stims_flag:
            self.hukusuu_sigeki = SearchFromHukusuuSigeki(dataset=args.dataset, num_stims=args.num_stims)
            self.paraphrase = self.hukusuu_sigeki.get_paraphrase_hukusuu_sigeki()
            self.nayose = self.hukusuu_sigeki.get_nayose_hukusuu_sigeki()
            # 複数の刺激語バージョンにおける、正解と不正解のリスト
            self.dict_keywords = self.hukusuu_sigeki.get_dict()
            # 複数の刺激語バージョンにおける、採用する刺激語の数
            self.num_stims = args.num_stims
            # wikipediaの出現頻度とか共起頻度とか分かる（予定）
            self.toigo = self.hukusuu_sigeki.get_toigo()
            self.kankei = self.hukusuu_sigeki.get_kankei()
        else:
            pass

        self.multi_stims_flag = args.multi_stims_flag


    def __call__(self, results_csv, results_csv_attention):
        results = []
        results_attention_and_raw = []
        # 複数→1つバージョンでは、answerは正解の連想語、keywordsは刺激語のリストのリスト
        # (謝罪) 1つ→複数の頃の名残でanswerという変数名だけど、複数→1つでは正解の連想語が入ります...
        for sid, (answer, keywords) in enumerate(self.dict_keywords.items()):
            if self.multi_stims_flag:
                # 連想文(str型)を作成する(この段階では刺激語はまだ入っていない)
                # input_sentencesはlist型
                input_sentences = self.keywords_to_sentences(keywords=keywords, human_word=answer)
            else:
                pass

            # 連想語頻度表から単語を持ってくる.
            if self.multi_stims_flag:
                # 相馬が決定した87題の中には、同じ正解語が含まれる(例：魚)ので、
                # 辞書のキーが魚、魚_2、魚_2_3のようになっている
                # 正解を判定する際には「_2」や「_2_3」は取り除かれる
                human_words = [answer]
            else:
                pass
            print(input_sentences)

            # 1つのキーワードで複数の入力文があるため, for文で愚直に回す
            for i, input_sentence in enumerate(input_sentences):
                # 連想
                # association_words_rawsは出力単語そのまま(変数名のrawは"生"。ローマ字で変数名を付けるのはやめよう！)
                # association_words_scoreは単語のスコア(おそらく、softmaxに通す前のlogits)
                # attenion_resultはattention(一応、全てのTransformer層から出力してあるはず...)
                association_words_raws, association_score_raws, attention_result = self.association(input_sentence, human_words)

                # ここのfor文で出力単語そのままではなく、刺激語を除いたり名詞だけ抽出したりする
                for association_words_raw, association_score_raw in zip(association_words_raws, association_score_raws):
                    # 出力単語のうち、名詞だけ抽出する
                    association_words, association_score, extract_num = self.extract_noun_from_output(association_words_raw, association_score_raw)

                    # 刺激語を除く
                    if self.multi_stims_flag:
                        for keyword in keywords:
                            association_words, association_score, extract_num = self.extract_paraphrase_from_output(keyword, human_words, association_words, association_score)
                    else:
                        pass

                    # 出力単語のうち、同じ単語を除く(名寄せ)
                    if self.args.output_nayose_flag:
                        association_words, association_score, extract_num = self.nayose_from_output(association_words, association_score)

                    # 結果を保存する
                    if self.multi_stims_flag:
                        result_list = [sid, keywords, input_sentence, human_words, self.toigo[human_words[0]], association_words, association_score]
                        tokenized_text = self.tokenizer.tokenize(input_sentence)
                        tokenized_sentence, _ = utils_tools.transform_tokenized_text_mecab_tohoku(tokenized_text=tokenized_text)                        
                        result_list_attentions_and_raws = [sid, keywords, input_sentence, tokenized_sentence, human_words, self.toigo[human_words[0]], attention_result, association_words_raw, association_score_raw]
                    else:
                        pass

                    results.append(result_list)
                    results_attention_and_raw.append(result_list_attentions_and_raws)

        header_results = ['sid', 'stims', 'input_sentence', 'answer', 'category', 'output_words', 'output_scores']       
        header_attns_and_raws = ['sid', 'stims', 'input_sentence', 'tokenized_sentence', 'answer', 'category', 'analyzed_attn', 'output_raw_words', 'output_raw_scores']

        # 結果を書き出す
        csv_writer(header=header_results, result=results, csv_file_path=results_csv)        

        # Attentionを書き出す
        csv_writer(header=header_attns_and_raws, result=results_attention_and_raw, csv_file_path=results_csv_attention)


    # 刺激語を入力文に変換する
    def keywords_to_sentences(self, keywords, human_word):
        """
        複数の刺激語バージョンにおける、連想文の作成
        """
        # 入力文を生成する
        input_sentences = []

        # 問い語(=限定語)を付与する場合
        if self.category_flag:
            if self.args.brackets_flag:
                # MASKに鍵括弧「」を付ける場合
                sentencess = utils_tools.hukusuu_sigeki_sentences_toigo_mask.items()
            else:
                # MASKに鍵括弧「」を付けない場合
                sentencess = utils_tools.hukusuu_sigeki_sentences_toigo.items()

            # %sは刺激語に置換する
            for number, sentences in sentencess:
                for toigo, parts in sentences.items():
                    if self.toigo[human_word] == toigo:
                        sentence_parts = []
                        for part in parts:
                            if part == "%s":
                                for i in range(self.args.num_stims):
                                    sentence_parts.append(keywords[i])
                                    if i == (self.args.num_stims - 1):
                                        pass
                                    else:
                                        sentence_parts.append("、")
                            else:
                                sentence_parts.append(part)
                        sentence = ''.join(sentence_parts)
                        input_sentences.append(sentence)

        # 問い語(=限定語)を付与しない場合
        else:
            if self.args.brackets_flag:
                # MASKに鍵括弧「」を付ける場合
                sentences = utils_tools.hukusuu_sigeki_sentences_mask.items()
            else:
                # MASKに鍵括弧「」を付けない場合
                sentences = utils_tools.hukusuu_sigeki_sentences.items()

            # %sは刺激語に置換する
            for number, parts in sentences:
                sentence_parts = []
                for part in parts:
                    if part == "%s":
                        for i in range(self.args.num_stims):
                            sentence_parts.append(keywords[i])
                            if i == (self.args.num_stims - 1):
                                pass
                            else:
                                sentence_parts.append("、")
                    else:
                        sentence_parts.append(part)
                sentence = ''.join(sentence_parts)
                input_sentences.append(sentence)

        return input_sentences


    # 連想を行い, 上位n位までのリストを返す関数
    def association(self, text, human_words):
        # textは 連想文
        # max_associationsは 出力する単語の数
        # human_words は 複数→1つバージョンでは正解の連想語

        # predictedsは、BERTの出力を格納するリスト
        predicteds = []
        predicted_tokens_list = []
        predicted_values_list_list = []

        if self.framework_opt == "tf":

            # テキストをトークナイザに入力(東北大学BERTの場合は、MeCab+WordPiece)
            tokenized_text = self.tokenizer.tokenize(text)

            # トークナイズした連想文に[MASK]や[CLS]、[SEP]を追加する + MASKの位置(masked_index)を出力
            # (本当はtokenizer.encoder_plusを使うと一発なのだが、実装した時は知らなかったのでそのままにしてある)
            tokenized_text, masked_index = utils_tools.transform_tokenized_text_mecab_tohoku(tokenized_text=tokenized_text)

            # トークンをIDに変換する
            tokenized_id = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            if self.model_opt == "cl-tohoku" or self.model_opt == "addparts":

                # 入力形式を整える
                tokenized_tensor = tf.constant(tokenized_id)
                tokenized_tensor = tf.expand_dims(tokenized_tensor, 0)

                # BERTモデルで出力。詳細は huggingface の TFBertForMaskedLM を要チェックだ！
                # output_attentions を True にすると Attention値が出力される
                # (Attention値はおそらくAttention Weightだけど、不安なので huggingface のソースコードを確認してほしい)
                outputs = self.model(tokenized_tensor, output_attentions=True)

                # predictionsは MaskedLMの出力で、各トークンに全語彙のスコアが付いたもの (次元は 1 × 入力長 × 語彙数)
                predictions = outputs[0]
                attentions = outputs[-1]

                # 予想トップnを出力する
                # valuesでスコアを確認できる(ソフトマックスに通す前のスコア、TFBertForMaskedLMのリファレンスで確認済み)
                for index in masked_index:
                    predicteds.append(tf.math.top_k(predictions[0, index], k=self.args.max_words))

            elif self.model_opt == "gmlp":
                # 多分 tf.constant の長さを 128 にしないといけない
                tokenized_tensor = tf.constant(tokenized_id)
                zeros = tf.zeros([128 - len(tokenized_tensor)], tf.int32)
                input_ids = tf.concat([tokenized_tensor, zeros], 0)
                input_ids = tf.reshape(input_ids, shape=(1, 128))
                # print(input_ids)
                input_mask = tf.constant(0, dtype=tf.int32, shape=[1, 128])
                input_type_ids = tf.constant(0, dtype=tf.int32, shape=[1, 128])
                temp_list = []
                for i in range(128):
                    if i < len(masked_index):
                        temp_list.append(masked_index[i])
                    else:
                        temp_list.append(0)
                masked_lm_positions = tf.constant(
                    [temp_list], dtype=tf.int32)
                inputs = [input_ids, input_mask, input_type_ids, masked_lm_positions]
                #print(inputs)
                outputs = self.model.predict(inputs)
                output = outputs['masked_lm']
                for i in range(len(masked_index)):
                    predicteds.append(tf.math.top_k(output[0][i], k=self.args.max_words))
                attentions = None

        for predicted in predicteds:
            # indexesで単語のインデックスを確認できる
            predicted_indexes = predicted.indices
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indexes)
            predicted_tokens_list.append(predicted_tokens)
            # スコアはTensor型なので, ndarrayに変換してからリストに変換している
            predicted_values = predicted.values
            predicted_values_ndarray = predicted_values.numpy()
            predicted_values_list = predicted_values_ndarray.tolist()
            predicted_values_list_list.append(predicted_values_list)

        # attentionsの分析を行う
        if self.model_opt != "gmlp":
            attention_result = self.analysis_attentions(attentions, tokenized_text, masked_index, human_words)
        else:
            attention_result = []

        return predicted_tokens_list, predicted_values_list_list, attention_result


    # attentionsの分析を行う関数(途中なので、ここを改造してほしい)
    def analysis_attentions(self, attentions, tokenized_text, masked_index, human_words):
        if self.multi_stims_flag:
            # 分析対象のTransformer層(-1は最終層、この数値はconfigで変更できるようにした方がいい)
            transformer_layers = [self.args.target_layer]
            # Attention_Headの数(本当はBERTのconfig.jsonを参照した方がいい)
            attention_head_num = 12
            """
                transformer_layer番目の層を取り出す
                attentions[transformer_layer]
                attention_head番目の層を取り出す
                attentions[transformer_layer][0][attention_head]
                maskからのattentionを取り出す
                attentions[transformer_layer][0][attention_head][masked_index]
                maskからの該当単語へのattentionを取り出す
                attentions[transformer_layer][0][attention_head][masked_index][word_index]
            """
            attention_result = None
            return attention_result


    # 出力から名詞を除く関数
    def extract_noun_from_output(self, association_words, association_score):

        # 単語のリストとスコアのリストを同時に更新する
        association_words_extract_noun = []
        association_score_extract_noun = []

        for i, association_word in enumerate(association_words):
            # [UNK]トークン と ## トークンを除く
            if association_word == "[UNK]" or "##" in association_word:
                continue
            # ginzaで処理する場合
            if self.extract_noun_opt == "ginza":
                doc = self.ginza(association_word)
                # print(doc)
                if doc[0].pos_ == "NOUN" or doc[0].pos_ == "PROPN":
                    association_words_extract_noun.append(association_word)
                    association_score_extract_noun.append(association_score[i])
            # mecabで処理する場合
            elif self.extract_noun_opt == "mecab":
                # mecab + 東北大学バージョン
                if (self.model_opt == "cl-tohoku") or (self.model_opt == "addparts") or (self.model_opt == "gmlp"):
                    nouns = [line.split()[0] for line in self.mecab.parse(association_word).splitlines()
                             if "名詞" in line.split()[-1]]
                    if nouns != []:
                        association_words_extract_noun.append(nouns[0])
                        association_score_extract_noun.append(association_score[i])
                else:
                    pass

        return association_words_extract_noun, association_score_extract_noun, len(association_words_extract_noun)


    # 刺激語や限定語を除く関数
    def extract_paraphrase_from_output(self, keyword, human_words, association_words, association_score):

        # 単語のリストとスコアのリストを同時に更新する
        association_words_extract_paraphrase = []
        association_score_extract_paraphrase = []

        for i, association_word in enumerate(association_words):
            # 連想文の〇〇、〇〇、〇〇...の都道府県は？の都道府県部分を除く
            if self.multi_stims_flag:
                if association_word == self.toigo[human_words[0]]:
                    continue
                elif self.toigo[human_words[0]] == "都道府県" and association_word == "県":
                    continue
                elif self.toigo[human_words[0]] == "家の中である場所" and association_word == "家":
                    continue

            if association_word in self.paraphrase[keyword]:
                continue
            else:
                association_words_extract_paraphrase.append(association_word)
                association_score_extract_paraphrase.append(association_score[i])

        return association_words_extract_paraphrase, association_score_extract_paraphrase, len(association_words_extract_paraphrase)


    # 出力を名寄せする関数
    def nayose_from_output(self, association_words, association_score):

        # 単語のリストとスコアのリストを同時に更新する
        association_words_nayose = []
        association_score_nayose = []

        for i, association_word in enumerate(association_words):

            # 名寄せの候補に含まれる場合，keyをリストに登録する
            # ここで名寄せしている
            for nayose_key, nayose_words in self.nayose.items():
                if association_word in nayose_words:
                    association_word = nayose_key
                    break

            # リストに既に同じ名詞がある場合はパス
            if association_word in association_words_nayose:
                pass
            else:
                association_words_nayose.append(association_word)
                association_score_nayose.append(association_score[i])

        return association_words_nayose, association_score_nayose, len(association_words_nayose)    
    

### 実験 ###

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Setting configurations")
    parser.add_argument('--output_words_from_bert', default=True, type=strtobool, help='Output words from BERT or not')
    parser.add_argument('--analysis_flag', default=True, type=strtobool, help='Analyze or not')
    parser.add_argument('--framework_opt', default='tf', type=str, help='Specify a framework')
    parser.add_argument('--model_opt', default='cl-tohoku', type=str, help='Specify a BERT model')
    parser.add_argument('--brackets_flag', default=True, type=strtobool, help='Adding brackets or not')
    parser.add_argument('--output_nayose_flag', default=True, type=strtobool, help='Nayose or not')
    parser.add_argument('--extract_noun_opt', default='mecab', type=str, help='[mecab, ginza]')
    parser.add_argument('--multi_stims_flag', default=True, type=strtobool, help='Version of stimulating')
    parser.add_argument('--category_flag', default=True, type=strtobool, help='Using categorizing word or not')
    parser.add_argument('--num_stims', default=5, type=int, help='number of stimulating words')
    parser.add_argument('--eval_opt', default='p', type=str, help='[p, MRR]')
    parser.add_argument('--ps', default=[1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 150], type=list, help='Specify ranks for analysis')
    parser.add_argument('--dataset', default='extract_keywordslist', type=str, help='dataset')
    parser.add_argument('--max_words', default=150, type=int, help='num of words from BERT')
    parser.add_argument('--another_analysis', default=293, type=int, help='Specify another method of analysis')
    parser.add_argument('--target_layer', default=-1, type=int, help='Specify output layer of transformer')
    parser.add_argument('--dict_mecab', default='unidic_lite', type=str, help='[unidic_lite, unidic, ipadic]')
    args = parser.parse_args()

    # bert_associationをインスタンス化
    bert_association = BertAssociation(args)

    # 分析バージョン
    # analysis_version = "human5_nayose_disred_output_nayose"
    
    save_dir = dir_name_getter(args)
    os.makedirs(save_dir, exist_ok=True)

    results_csv = save_dir + "result.csv"
    results_csv_attention = save_dir + "result_attentions_and_raws.csv"

    output1_csv = save_dir + "analysis.csv"
    output2_csv = save_dir + "analysis_analysis.csv"

    # 単語を出力する
    if args.output_words_from_bert:
        bert_association(results_csv=results_csv, results_csv_attention=results_csv_attention)

    # 集計する
    if args.analysis_flag:
        analyzer = Analyzer(args)
        if args.another_analysis == 0 or args.another_analysis == 293:
            analyzer.analysis_result_match_nayose(results_csv=results_csv, output_csv=output1_csv, bert_interval=1)
            analyzer.analysis_analysis_result_match(results_csv=output1_csv, output_csv=output2_csv)
            analyzer.hits_at_k(results_dir=save_dir, target_ranks=args.ps)