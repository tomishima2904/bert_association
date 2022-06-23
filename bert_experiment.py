# 必要なライブラリ。 pipやcondaでインストールしてください。
from unicodedata import category
from transformers import BertJapaneseTokenizer, TFBertForMaskedLM, BertConfig
import tensorflow as tf
import spacy
import os
import pandas as pd
import fugashi
import datetime
import io,sys

# 自作ファイルからのインポート
sys.path.append('.')
from dict_maker import SearchBasedOnStimulations
import utils_tools
from models import Model
from analysis2 import Analyzer2
from file_handler import *
from config import args

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # 日本語を取り扱う時はとりあえずこれを書いておくといいかも


# B4の頃から違法増築を繰り返した結果、一つのクラスに色々詰め込むゴミクラスが完成した...
class BertAssociation():
    # kwargsとか使った方がいい...
    def __init__(self, args):

        self.args = args  # configやコマンドライン引数から設定した引数を渡す

        if self.args.reverse_flag: assert self.args.category_opt=='cat'  # category_optが 'cat' じゃないとrevをできないようにする
        if self.args.reverse_flag: self.target_mask = '{cat}'  # 'rev' の場合カテゴリーを答えさせるので utils_tools にある連想文の {cat} が[MASK]になる
        else: self.target_mask = '{ans}'  # 'rev' 以外なら utils_tools にある連想文の {ans} が[MASK]になる

        mecab_option = f"-d {self.args.packages_path}/{self.args.dict_mecab}/dicdir -r {self.args.packages_path}/{self.args.dict_mecab}/dicdir/mecabrc"  # 名詞判定で使用するmecabの設定

        if args.framework_opt == "tf":
            print(tf.__version__)
            model_object = Model(mecab_option, args)
            self.tokenizer = model_object.tokenizer
            self.model = model_object.model
            self.mecab = model_object.mecab

        self.ginza = spacy.load('ja_ginza')  # spacy の ginza モデルをインストール

        if args.multi_stims_flag:
            self.search_based_on_stimulations = SearchBasedOnStimulations(args)
            self.paraphrase = self.search_based_on_stimulations.get_paraphrase_hukusuu_sigeki()
            self.nayose = self.search_based_on_stimulations.get_nayose_hukusuu_sigeki()
            # 複数の刺激語バージョンにおける、正解と不正解のリスト
            self.dict_keywords = self.search_based_on_stimulations.get_dict()

            # wikipediaの出現頻度とか共起頻度とか分かる（予定）
            # self.toigo = self.search_based_on_stimulations.get_toigo()
            self.kankei = self.search_based_on_stimulations.get_kankei()
        else:
            pass

        if args.category_opt=='cat':  # カテゴリー指定有の場合
            self.categories_and_sentences = utils_tools.base_sentences_and_synonyms
        elif args.category_opt=='WOcat':  # カテゴリー指定無の場合(論文でのカテゴリー指定無との使われ方が違うので注意！)
            self.base_sentence = utils_tools.base_sentence_without_category
        elif args.category_opt=='kotoba':  # 「言葉」とする場合(論文でのカテゴリー指定無はこちら)
            self.base_sentence = utils_tools.base_sentence
        else:  # 上記以外の文字列を受けとった場合はエラー
            print('category option error\n')
            sys.exit()


    def __call__(self, results_dir):
        results = []  # result_xx.csvとして出力する結果
        results_attention_and_raw = []  # result_attentions_and_raws_xx.csvとして出力する結果
        results_summary = []  # result_summary_xx.csvとして出力する結果
        results_all_summary = []  # results_all_summaryとして出力する結果

        # 複数→1つバージョンでは、answerは正解の連想語、keywordsは刺激語のリストのリスト
        for sid, values in self.dict_keywords.items():
            if args.multi_stims_flag:
                # 連想文(str型)を作成する(この段階では刺激語はまだ入っていない)
                # input_sentencesはlist型
                if args.category_opt=='cat':
                    base_sentence = self.categories_and_sentences[values['category']]
                    if  ( '{cat}' not in base_sentence['sentence']) and self.args.reverse_flag: continue
                    else: input_sentences, category_synonyms = self._keywords_to_sentences(values=values)
                else:
                    input_sentences = self._keywords_to_sentences(values=values)
                    category_synonyms = ['']*len(input_sentences)

            else:
                pass

            # 連想語頻度表から単語を持ってくる.
            if args.multi_stims_flag:
                human_words = [values['answer']]  # 正解語を設定する  str->list
            else:
                pass
            print(input_sentences)


            # 1つのキーワードで複数の入力文があるため, for文で愚直に回す
            for i, input_sentence in enumerate(input_sentences):
                # 連想
                # 生の出力語, その出力語のlogitsの値, 全層全ヘッドのattention weight
                association_words_raws, association_score_raws, attention_result = self.association(input_sentence, human_words)

                # ここのfor文で出力単語そのままではなく、刺激語を除いたり名詞だけ抽出したりする
                for association_words_raw, association_score_raw in zip(association_words_raws, association_score_raws):
                    # 出力単語のうち、名詞だけ抽出する
                    association_words, association_score, extract_num = self._extract_noun_from_output(association_words_raw, association_score_raw)

                    # 刺激語を除く
                    if args.multi_stims_flag:
                        for stim in values['stims']:
                            association_words, association_score, extract_num = self._extract_paraphrase_from_output(stim, values, association_words, association_score)
                    else:
                        pass

                    # 出力単語のうち、同じ単語を除く(名寄せ)
                    if self.args.output_nayose_flag:
                        association_words, association_score, extract_num = self._nayose_from_output(association_words, association_score)

                    # 結果を保存する
                    if args.multi_stims_flag:
                        if args.category_opt: sentence_id = f'{sid:0>3}-{i:0>2}'
                        else: sentence_id = f'{sid:0>3}-'
                        result_list = [sentence_id, values['stims'], input_sentence, human_words, values['category'], category_synonyms[i], association_words, association_score]
                        tokenized_text = self.tokenizer.tokenize(input_sentence)
                        tokenized_sentence, _ = utils_tools.transform_tokenized_text_mecab_tohoku(tokenized_text=tokenized_text)
                        result_list_attentions_and_raws = [sentence_id, values['stims'], input_sentence, tokenized_sentence, human_words, values['category'], category_synonyms[i], attention_result, association_words_raw, association_score_raw]
                        if self.args.category_opt == 'cat' and self.args.reverse_flag == False:
                            results_all_summary.append([sentence_id, values['category'], category_synonyms[i], human_words, input_sentence, association_words[:self.args.summary_num]])
                            if i==0:
                                results_summary.append([sentence_id, values['category'], human_words, input_sentence, association_words[:self.args.summary_num]])
                        else:
                            results_summary.append([sentence_id, values['category'], human_words, input_sentence, association_words[:self.args.summary_num]]  )
                    else:
                        pass

                    results.append(result_list)
                    results_attention_and_raw.append(result_list_attentions_and_raws)


        header_results = ['sid', 'stims', 'input_sentence', 'answer', 'category', 'category_synonyms', 'output_words', 'output_scores']
        header_attns_and_raws = ['sid', 'stims', 'input_sentence', 'tokenized_sentence', 'answer', 'category', 'category_synonyms', 'attn_weights_of_mask', 'output_raw_words', 'output_raw_scores']
        header_summary = ['sid', 'category', 'answer', 'input_sentence', 'output_words']

        results_csv = f'{results_dir}/result_{file_name_getter(self.args)}.csv'
        results_csv_attention = f'{results_dir}/result_attentions_and_raws_{file_name_getter(self.args)}.csv'
        results_csv_summary = f'{results_dir}/result_summary_{self.args.summary_num}_{file_name_getter(self.args)}.csv'

        # 結果を書き出す
        csv_writer(header=header_results, result=results, csv_file_path=results_csv)

        # Attentionを書き出す
        csv_writer(header=header_attns_and_raws, result=results_attention_and_raw, csv_file_path=results_csv_attention)

        # Summaryを書き出す
        csv_writer(header=header_summary, result=results_summary, csv_file_path=results_csv_summary)
        if self.args.category_opt == 'cat' and self.args.reverse_flag == False:
            header_all_summary = ['sid', 'category', 'synonyms', 'answer', 'input_sentence', 'output_words']
            results_csv_all_summary = f'{results_dir}/result_all_summary_{self.args.summary_num}_{file_name_getter(self.args)}.csv'
            csv_writer(header=header_all_summary, result=results_all_summary, csv_file_path=results_csv_all_summary)


    # 刺激語を入力文に変換する
    def _keywords_to_sentences(self, values):
        """
        複数の刺激語バージョンにおける、連想文の作成
        """
        # 入力文を生成する
        input_sentences = []
        category_synonyms = []

        # カテゴリー指定有の場合
        if args.category_opt=='cat':
            # {stims}は刺激語に置換する
            base_sentence = self.categories_and_sentences[values['category']]  # base_sentenceは元となる文
            # カテゴリー指定語が単語で表せない場合
            if len(base_sentence['synonyms']) == 0:
                sentence_parts = []
                for part in base_sentence['sentence']:
                    if part == self.target_mask:  # [MASK]に置換
                        if self.args.brackets_flag:  sentence_parts.append('「[MASK]」')  # 「」有
                        else: sentence_parts.append('[MASK]')  # 「」無
                    elif part == "{stims}":
                        sentence_parts.append('、'.join(values['stims']))  # 連想文の {stims} を刺激語を置換
                    elif part == '{ans}':
                        sentence_parts.append(values['answer'])  # 連想文の {ans} を正解語を置換
                    elif part == '{cat}':
                        sentence_parts.append(values['category'])  # 連想文の {cat} をカテゴリー指定語を置換
                    else:
                        sentence_parts.append(part)
                category_synonyms.append(values['category'])
                sentence = ''.join(sentence_parts)
                input_sentences.append(sentence)
            # カテゴリー指定語が単語で表せる場合
            else:
                for synonym in base_sentence['synonyms']:
                    sentence_parts = []
                    for part in base_sentence['sentence']:
                        if part == self.target_mask:  # [MASK]に置換
                            if self.args.brackets_flag:  sentence_parts.append('「[MASK]」')  # 「」有
                            else: sentence_parts.append('[MASK]')  # 「」無
                        elif part == "{stims}":
                            sentence_parts.append('、'.join(values['stims']))  # 連想文の {stims} を刺激語を置換
                        elif part =='{ans}':
                            sentence_parts.append(values['answer'])  # 連想文の {ans} を正解語を置換
                        elif part == '{cat}':
                            sentence_parts.append(synonym)  # 文中に {cat} は存在しないので意味のない文かも
                        else:
                            sentence_parts.append(part)

                    sentence = ''.join(sentence_parts)
                    input_sentences.append(sentence)
                    if self.args.reverse_flag:
                        category_synonyms.append(values['category'])
                        return input_sentences, category_synonyms
                    else:
                        if len(synonym) == 0: category_synonyms.append('0nan')
                        else: category_synonyms.append(synonym)

            return input_sentences, category_synonyms  # 入力文, カテゴリー指定語の類似語

        # カテゴリー指定有以外の場合
        else:
            sentence_parts = []
            for part in self.base_sentence:
                if part == self.target_mask:  # [MASK]に置換
                    if self.args.brackets_flag:  sentence_parts.append('「[MASK]」')  # 「」有
                    else: sentence_parts.append('[MASK]')  # 「」無
                elif part == "{stims}":
                    sentence_parts.append('、'.join(values['stims']))  # 連想文の {stims} を刺激語を置換
                else:
                    sentence_parts.append(part)
            sentence = ''.join(sentence_parts)
            input_sentences.append(sentence)

            return input_sentences  # 入力文


    # 連想を行い, 上位n位までのリストを返す関数
    def association(self, text, human_words):
        # textは 連想文
        # max_associationsは 出力する単語の数
        # human_words は 複数→1つバージョンでは正解の連想語

        # predictedsは、BERTの出力を格納するリスト
        predicteds = []
        predicted_tokens_list = []
        predicted_values_list_list = []

        if args.framework_opt == "tf":

            # テキストをトークナイザに入力(東北大学BERTの場合は、MeCab+WordPiece)
            tokenized_text = self.tokenizer.tokenize(text)

            # トークナイズした連想文に[MASK]や[CLS]、[SEP]を追加する + MASKの位置(masked_indiceis)を出力
            # (本当はtokenizer.encoder_plusを使うと一発なのだが、実装した時は知らなかったのでそのままにしてある)
            tokenized_text.insert(0, '[CLS]')
            tokenized_text[-1] = '[SEP]'
            masked_indicies = [tokenized_text.index(token) for token in tokenized_text if token == '[MASK]']
            # tokenized_text, masked_indiceis = utils_tools.transform_tokenized_text_mecab_tohoku(tokenized_text=tokenized_text)

            # トークンをIDに変換する
            tokenized_id = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            if args.model_opt == "cl-tohoku" or args.model_opt == "addparts":

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
                for index in masked_indicies:
                    predicteds.append(tf.math.top_k(predictions[0, index], k=self.args.max_words))

            elif args.model_opt == "gmlp":
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
                    if i < len(masked_indicies):
                        temp_list.append(masked_indicies[i])
                    else:
                        temp_list.append(0)
                masked_lm_positions = tf.constant(
                    [temp_list], dtype=tf.int32)
                inputs = [input_ids, input_mask, input_type_ids, masked_lm_positions]
                #print(inputs)
                outputs = self.model.predict(inputs)
                output = outputs['masked_lm']
                for i in range(len(masked_indicies)):
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
        if args.model_opt != "gmlp":
            attention_result = self._analysis_attentions(attentions, masked_indicies)
        else:
            attention_result = []

        return predicted_tokens_list, predicted_values_list_list, attention_result


    # attentionsの分析を行う関数(途中なので、ここを改造してほしい)
    def _analysis_attentions(self, attentions, masked_indicies):
        if args.multi_stims_flag:
            attention_result = []
            transformer_layer = attentions[self.args.target_layer]  # tatget_layer番目の層のattentionを取り出す
            attention_head_num = 12  # Attention_Headの数(本当はBERTのconfig.jsonを参照した方がいい)
            # if self.args.target_heads == None:
            for head in range(attention_head_num):
                attn_of_mask = transformer_layer[0][head][masked_indicies[0]]  # 対象のattention-headのattention-weightを取り出す
                attention_result.append(attn_of_mask.numpy().tolist())

            # else:
            #     attn_of_mask = transformer_layer[0][self.args.target_heads][masked_indicies[0]]
            #     attention_result.append(attn_of_mask.numpy().tolist())

            return attention_result


    # 出力から名詞を除く関数
    def _extract_noun_from_output(self, association_words, association_score):

        # 単語のリストとスコアのリストを同時に更新する
        association_words_extract_noun = []
        association_score_extract_noun = []

        for i, association_word in enumerate(association_words):
            # [UNK]トークン と ## トークンを除く
            if association_word == "[UNK]" or "##" in association_word:
                continue
            # ginzaで処理する場合
            if args.extract_noun_opt == "ginza":
                doc = self.ginza(association_word)
                # print(doc)
                if doc[0].pos_ == "NOUN" or doc[0].pos_ == "PROPN":
                    association_words_extract_noun.append(association_word)
                    association_score_extract_noun.append(association_score[i])
            # mecabで処理する場合
            elif args.extract_noun_opt == "mecab":
                # mecab + 東北大学バージョン
                if (args.model_opt == "cl-tohoku") or (args.model_opt == "addparts") or (args.model_opt == "gmlp"):
                    nouns = [line.split()[0] for line in self.mecab.parse(association_word).splitlines()
                             if "名詞" in line.split()[-1]]
                    if nouns != []:
                        association_words_extract_noun.append(nouns[0])
                        association_score_extract_noun.append(association_score[i])
                else:
                    pass

        return association_words_extract_noun, association_score_extract_noun, len(association_words_extract_noun)


    # 刺激語や限定語を除く関数
    def _extract_paraphrase_from_output(self, stim, values, association_words, association_score):

        # 単語のリストとスコアのリストを同時に更新する
        association_words_extract_paraphrase = []
        association_score_extract_paraphrase = []

        for i, association_word in enumerate(association_words):
            # 連想文の〇〇、〇〇、〇〇...の都道府県は？の都道府県部分を除く
            if args.multi_stims_flag:
                if not self.args.reverse_flag:
                    if association_word == values['category']:
                        continue
                    elif values['category'] == "都道府県" and association_word == "県":
                        continue
                    elif values['category'] == "家の中である場所" and association_word == "家":
                        continue

                else:
                    if association_word == values['answer']:
                        continue

            if association_word in self.paraphrase[stim]:
                continue
            else:
                association_words_extract_paraphrase.append(association_word)
                association_score_extract_paraphrase.append(association_score[i])

        return association_words_extract_paraphrase, association_score_extract_paraphrase, len(association_words_extract_paraphrase)


    # 出力を名寄せする関数
    def _nayose_from_output(self, association_words, association_score):

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

    # bert_associationをインスタンス化
    bert_association = BertAssociation(args)

    # 分析バージョン
    # analysis_version = "human5_nayose_disred_output_nayose"

    save_dir = dir_name_getter(args)
    os.makedirs(save_dir, exist_ok=True)

    results_csv = save_dir + f"/result_{file_name_getter(args)}.csv"
    results_csv_attention = save_dir + f"/result_attentions_and_raws_{file_name_getter(args)}.csv"

    output1_csv = save_dir + f"/analysis_{file_name_getter(args)}.csv"
    output2_csv = save_dir + f"/analysis_analysis_{file_name_getter(args)}.csv"

    # 単語を出力する
    if args.output_words_from_bert:
        bert_association(save_dir)

    # 集計する
    if args.analysis_flag:
        analyzer = Analyzer2(args)
        if args.another_analysis == 0 or args.another_analysis == 293:
            analyzer.analysis_result_match_nayose(results_csv=results_csv, output_csv=output1_csv, bert_interval=1)
            # analyzer.analysis_analysis_result_match(results_csv=output1_csv, output_csv=output2_csv)
            analyzer.hits_at_k(results_dir=save_dir, target_ranks=args.ps)
            analyzer.attention_visualizasion(save_dir)