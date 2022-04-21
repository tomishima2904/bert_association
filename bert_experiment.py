# 必要なライブラリ。 pipやcondaでインストールしてください。
from transformers import BertJapaneseTokenizer, TFBertForMaskedLM, BertConfig
import tensorflow as tf
import spacy
import csv
import ast
import os
import pandas as pd
import fugashi

# 自作ファイルからのインポート
from extract_hukusuu import SearchFromHukusuuSigeki
import utils_tools


# B4の頃から違法増築を繰り返した結果、一つのクラスに色々詰め込むゴミクラスが完成した...
class BertAssociation():
    # kwargsとか使った方がいい...
    def __init__(self, framework_flag=None, model_flag=None, sigeki_num=None, hukusuu_sigeki_flag=None, hukusuu_sigeki_dataset=None, toigo_flag=None, extract_verb_flag=None):

        # ～flagと書いてあるけど、いくつかbool型じゃなくてstr型なので注意！
        # flagと書いてあるけどstr型その1。モデルを定める。
        self.model_flag = model_flag

        # flagと書いてあるけどstr型その2。使用するフレームワークを定める(学習しないのであまり区別する意味はないが...)
        self.framework_flag = framework_flag

        # 問い語(=限定語)を使用するかどうか。
        # Trueの場合は「都道府県は～」「色は～」という連想文になる。
        # Falseの場合は「言葉は～」という連想文になる。
        # utils_tools.pyに連想文を書いてある。
        self.toigo_flag = toigo_flag

        # flagと書いてあるけどstr型その3。名詞を抽出する際に使用するライブラリ？を定める
        # mecab...MeCabを使用する
        # ginza...Ginzaを使用する
        self.extract_verb_flag = extract_verb_flag

        # 名詞判定で使用するmecabの設定(環境に合わせて設定してね。これは相馬のローカル環境の例)
        mecab_option = "-d C:/Users/yuya1/Anaconda3/envs/tensorflow-labo/lib/site-packages/ipadic/dicdir -r C:/Users/yuya1/Anaconda3/envs/tensorflow-labo/lib/site-packages/ipadic/dicdir/mecabrc"

        if self.framework_flag == "tf":
            print(tf.__version__)
            if self.model_flag == "cl-tohoku":
                # 東北大学の乾研究室が作成した日本語BERTモデル
                model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
                self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
                self.model = TFBertForMaskedLM.from_pretrained(model_name, from_pt=True)
                # Mecab
                self.mecab = fugashi.GenericTagger(mecab_option)
            elif self.model_flag == "addparts":
                # 東北大学の乾研究室が作成した日本語BERTモデルを元に、名詞だけを事前学習したBERT
                config = BertConfig.from_json_file('addparts/config.json')
                self.tokenizer = BertJapaneseTokenizer.from_pretrained('addparts/vocab.txt', do_lower_case=False, word_tokenizer_type="mecab", mecab_dic_type='unidic_lite', unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]')
                self.model = TFBertForMaskedLM.from_pretrained('addparts/pytorch_model.bin', config=config, from_pt=True)
                # Mecab
                self.mecab = fugashi.GenericTagger(mecab_option)
            elif self.model_flag == "gmlp":
                from official.nlp.modeling.networks import gmlp_encoder_remake
                from official.nlp.modeling import models
                from official.modeling import tf_utils
                from official.nlp.gmlp import configs

                transformer_encoder = gmlp_encoder_remake.GmlpEncoder(
                    vocab_size=32768,
                    hidden_size=768,
                    num_layers=24,
                    sequence_length=128,
                    activation=tf.nn.swish,
                    initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                    return_all_encoder_outputs=False,
                    output_range=None,
                    embedding_width=None,
                    embedding_layer=None,
                    max_sequence_length=None,
                    type_vocab_size=0,
                    num_attention_heads=0,
                    intermediate_size=0,
                    dropout_rate=0.0,
                    attention_dropout_rate=0.0)
                self.model = models.GmlpPretrainer(
                    network=transformer_encoder,
                    embedding_table=transformer_encoder.get_embedding_table(),
                    num_classes=2,  # The next sentence prediction label has two classes.
                    activation=tf_utils.get_activation(
                        configs.GmlpConfig.from_json_file("gMLP/config.json").hidden_act),
                    num_token_predictions=80,
                    output='logits')

                checkpoint = tf.train.Checkpoint(model=self.model)
                checkpoint.restore("gMLP/gmlp_model_step_2000000.ckpt-5")

                self.tokenizer = BertJapaneseTokenizer.from_pretrained('gMLP/vocab.txt', do_lower_case=False,
                                                                       word_tokenizer_type="mecab",
                                                                       mecab_dic_type='unidic_lite', unk_token='[UNK]',
                                                                       sep_token='[SEP]', pad_token='[PAD]',
                                                                       cls_token='[CLS]', mask_token='[MASK]')
                self.mecab = fugashi.GenericTagger(mecab_option)

        # spacy の ginza モデルをインストール
        self.ginza = spacy.load('ja_ginza')

        if hukusuu_sigeki_flag:
            self.hukusuu_sigeki = SearchFromHukusuuSigeki(dataset=hukusuu_sigeki_dataset)
            self.paraphrase = self.hukusuu_sigeki.get_paraphrase_hukusuu_sigeki()
            self.nayose = self.hukusuu_sigeki.get_nayose_hukusuu_sigeki()
            # 複数の刺激語バージョンにおける、正解と不正解のリスト
            self.dict_keywords = self.hukusuu_sigeki.get_dict()
            # 複数の刺激語バージョンにおける、採用する刺激語の数
            self.sigeki_num = sigeki_num
            # wikipediaの出現頻度とか共起頻度とか分かる（予定）
            self.toigo = self.hukusuu_sigeki.get_toigo()
            self.kankei = self.hukusuu_sigeki.get_kankei()
        else:
            pass

        self.hukusuu_sigeki_flag = hukusuu_sigeki_flag

    def __call__(self, results_csv, results_csv_attention, max_association, mask_kaxtuko_flag, output_nayose_flag):
        results = []
        results_attention_and_nama = []
        # 複数→1つバージョンでは、colorは正解の連想語、keywordsは刺激語のリストのリスト
        # (謝罪) 1つ→複数の頃の名残でcolorという変数名だけど、複数→1つでは正解の連想語が入ります...
        for color, keywords in self.dict_keywords.items():
            if self.hukusuu_sigeki_flag:
                # 連想文(str型)を作成する(この段階では刺激語はまだ入っていない)
                # input_sentencesはlist型
                input_sentences = self.keywords_to_sentences(keywords=keywords, sigeki_num=self.sigeki_num, human_word=color, mask_kaxtuko_flag=mask_kaxtuko_flag)
            else:
                pass

            # 連想語頻度表から単語を持ってくる.
            if self.hukusuu_sigeki_flag:
                # 相馬が決定した87題の中には、同じ正解語が含まれる(例：魚)ので、
                # 辞書のキーが魚、魚_2、魚_2_3のようになっている
                # 正解を判定する際には「_2」や「_2_3」は取り除かれる
                human_words = [color]
            else:
                pass
            print(input_sentences)

            # 1つのキーワードで複数の入力文があるため, for文で愚直に回す
            for i, input_sentence in enumerate(input_sentences):
                # 連想
                # association_words_namasは出力単語そのまま(変数名のnamaは"生"。ローマ字で変数名を付けるのはやめよう！)
                # association_words_scoreは単語のスコア(おそらく、softmaxに通す前のlogits)
                # attenion_resultはattention(一応、全てのTransformer層から出力してあるはず...)
                association_words_namas, association_score_namas, attention_result = self.association(input_sentence, max_association, human_words)

                # ここのfor文で出力単語そのままではなく、刺激語を除いたり名詞だけ抽出したりする
                for association_words_nama, association_score_nama in zip(association_words_namas, association_score_namas):
                    # 出力単語のうち、名詞だけ抽出する
                    association_words, association_score, extract_num = self.extract_verb_from_output(association_words_nama, association_score_nama)

                    # 刺激語を除く
                    if self.hukusuu_sigeki_flag:
                        for keyword in keywords:
                            association_words, association_score, extract_num = self.extract_paraphrase_from_output(keyword, human_words, association_words, association_score)
                    else:
                        pass

                    # 出力単語のうち、同じ単語を除く(名寄せ)
                    if output_nayose_flag:
                        association_words, association_score, extract_num = self.nayose_from_output(association_words, association_score)

                    # 結果を保存する
                    if self.hukusuu_sigeki_flag:
                        result_list = [keywords, i, input_sentence, human_words, association_words, association_score]
                        result_list_attentions_and_namas = [keywords, i, input_sentence, human_words, attention_result, association_words_nama, association_score_nama]
                    else:
                        pass

                    results.append(result_list)
                    results_attention_and_nama.append(result_list_attentions_and_namas)

        # 結果を書き出す
        with open(results_csv, 'w', newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(results)

        # Attentionを書き出す
        with open(results_csv_attention, 'w', newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(results_attention_and_nama)

    # 刺激語を入力文に変換する
    def keywords_to_sentences(self, keywords, sigeki_num, human_word, mask_kaxtuko_flag):
        """
        複数の刺激語バージョンにおける、連想文の作成
        """
        # 入力文を生成する
        input_sentences = []

        # 問い語(=限定語)を付与する場合
        if self.toigo_flag:
            if mask_kaxtuko_flag:
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
                                for i in range(sigeki_num):
                                    sentence_parts.append(keywords[i])
                                    if i == (sigeki_num - 1):
                                        pass
                                    else:
                                        sentence_parts.append("、")
                            else:
                                sentence_parts.append(part)
                        sentence = ''.join(sentence_parts)
                        input_sentences.append(sentence)

        # 問い語(=限定語)を付与しない場合
        else:
            if mask_kaxtuko_flag:
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
                        for i in range(sigeki_num):
                            sentence_parts.append(keywords[i])
                            if i == (sigeki_num - 1):
                                pass
                            else:
                                sentence_parts.append("、")
                    else:
                        sentence_parts.append(part)
                sentence = ''.join(sentence_parts)
                input_sentences.append(sentence)

        return input_sentences

    # 連想を行い, 上位n位までのリストを返す関数
    def association(self, text, max_association, human_words):
        # textは 連想文
        # max_associationは 出力する単語の数
        # human_words は 複数→1つバージョンでは正解の連想語

        # predictedsは、BERTの出力を格納するリスト
        predicteds = []
        predicted_tokens_list = []
        predicted_values_list_list = []

        if self.framework_flag == "tf":

            # テキストをトークナイザに入力(東北大学BERTの場合は、MeCab+WordPiece)
            tokenized_text = self.tokenizer.tokenize(text)

            # トークナイズした連想文に[MASK]や[CLS]、[SEP]を追加する + MASKの位置(masked_index)を出力
            # (本当はtokenizer.encoder_plusを使うと一発なのだが、実装した時は知らなかったのでそのままにしてある)
            tokenized_text, masked_index = utils_tools.transform_tokenized_text_mecab_tohoku(tokenized_text=tokenized_text)

            # トークンをIDに変換する
            tokenized_id = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            if self.model_flag == "cl-tohoku" or self.model_flag == "addparts":

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
                    predicteds.append(tf.math.top_k(predictions[0, index], k=max_association))

            elif self.model_flag == "gmlp":
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
                    predicteds.append(tf.math.top_k(output[0][i], k=max_association))
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
        if self.model_flag != "gmlp":
            attention_result = self.analysis_attentions(attentions, tokenized_text, masked_index, human_words)
        else:
            attention_result = []

        return predicted_tokens_list, predicted_values_list_list, attention_result

    # attentionsの分析を行う関数(途中なので、ここを改造してほしい)
    def analysis_attentions(self, attentions, tokenized_text, masked_index, human_words):
        if self.hukusuu_sigeki_flag:
            # 分析対象のTransformer層(-1は最終層、この数値はconfigで変更できるようにした方がいい)
            transformer_layers = [-1]
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
    def extract_verb_from_output(self, association_words, association_score):

        # 単語のリストとスコアのリストを同時に更新する
        association_words_extract_verb = []
        association_score_extract_verb = []

        for i, association_word in enumerate(association_words):
            # [UNK]トークン と ## トークンを除く
            if association_word == "[UNK]" or "##" in association_word:
                continue
            # ginzaで処理する場合
            if self.extract_verb_flag == "ginza":
                doc = self.ginza(association_word)
                # print(doc)
                if doc[0].pos_ == "NOUN" or doc[0].pos_ == "PROPN":
                    association_words_extract_verb.append(association_word)
                    association_score_extract_verb.append(association_score[i])
            # mecabで処理する場合
            elif self.extract_verb_flag == "mecab":
                # mecab + 東北大学バージョン
                if (self.model_flag == "cl-tohoku") or (self.model_flag == "addparts") or (self.model_flag == "gmlp"):
                    nouns = [line.split()[0] for line in self.mecab.parse(association_word).splitlines()
                             if "名詞" in line.split()[-1]]
                    if nouns != []:
                        association_words_extract_verb.append(nouns[0])
                        association_score_extract_verb.append(association_score[i])
                else:
                    pass

        return association_words_extract_verb, association_score_extract_verb, len(association_words_extract_verb)

    # 刺激語や限定語を除く関数
    def extract_paraphrase_from_output(self, keyword, human_words, association_words, association_score):

        # 単語のリストとスコアのリストを同時に更新する
        association_words_extract_paraphrase = []
        association_score_extract_paraphrase = []

        for i, association_word in enumerate(association_words):
            # 連想文の〇〇、〇〇、〇〇...の都道府県は？の都道府県部分を除く
            if self.hukusuu_sigeki_flag:
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

    # results_csvを読み込む関数
    def read_results_csv(self, results_csv):
        try:
            df = pd.read_csv(results_csv, header=None, encoding="utf-8", engine="python")
        except:
            df = pd.read_csv(results_csv, header=None, engine="python")
        print(df)
        results = df[[0, 1, 2, 3, 4, 5]]
        # [keyword, i, input_sentence, color, association_words, association_score]
        # 1...キーワード,
        # 2...入力文の番号,
        # 3...入力文,
        # 4...目的の色,
        # 5...出力された単語(150単語, 順位順, 出現しなかった場合は-1)
        # 6...出力された単語のスコア
        return results

    # analysis_result_matchで使用する書き込み関数
    def write_csv_match(self, result_csv, csv_file):
        with open(csv_file, 'w', newline="",  encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(result_csv)

    # 出力結果の分析
    def analysis_result_match_nayose(self, results_csv, output_csv, bert_interval, max_word_num, another_flag):
        results = self.read_results_csv(results_csv)

        # 出力した単語と正解の連想語が一致する場合にリストにぶち込んだりカウントを足したりする
        result_match = []
        for i, result in enumerate(results.itertuples()):
            if another_flag == 293:
                human_words = ast.literal_eval(result[4])

                for i, human_word in enumerate(human_words):
                    human_words[i] = ''.join([i for i in human_word if not i.isdigit()])

                # 出力された単語とスコアのリスト
                result_str_to_list_words = ast.literal_eval(result[5])
                result_str_to_list_score = ast.literal_eval(result[6])

                # 人間の連想語と一致した単語とそのスコア
                bert_and_human_word = []
                bert_and_human_score = []

                # 人間の連想語と一致しなかった単語とそのスコア
                not_bert_and_human_word = []
                not_bert_and_human_score = []

                # 人間の連想語と一致した数（bert_intervalごと）
                not_bert_and_human_num = []

                # 人間の連想語が出現した位置
                human_words_rank = 0

                # 単語1つ1つ走査する
                for j in range(max_word_num):
                    if j < len(result_str_to_list_words):
                        bert_word = result_str_to_list_words[j]

                        # BERTと人間の一致する単語
                        # 全ての連想語は名寄せしようがないので，名寄せバージョンではない
                        if bert_word in human_words:
                            human_words_rank = j+1
                            bert_and_human_word.append(bert_word)
                            bert_and_human_score.append(result_str_to_list_score[j])
                        else:
                            not_bert_and_human_word.append(bert_word)
                            not_bert_and_human_score.append(result_str_to_list_score[j])

                    # インターバルごとに記録する
                    if j % bert_interval == bert_interval-1:
                        not_bert_and_human_num.append(len(not_bert_and_human_word))

                result_tmp = i, result[1], result[2], result[3], result[4], human_words_rank, bert_and_human_word, bert_and_human_score, not_bert_and_human_word, not_bert_and_human_score, not_bert_and_human_num
                result_match.append(result_tmp)

        self.write_csv_match(result_match, output_csv)

    # スコアを算出する
    def analysis_analysis_result_match(self, results_csv, output_csv, another_flag, ps, eval_flag):
        df = pd.read_csv(results_csv, header=None, engine="python")
        results = df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        # 1...通し番号,
        # 2...キーワード,
        # 3...連想文の番号,
        # 4...連想文,
        # another_flag == 293
        # 5...人間の連想するはずの単語
        # 6...人間の連想した単語の順位
        # 7...人間と出力が一致した単語
        # 8...人間と出力が一致した単語のスコア,

        # 連想文の組み合わせ
        if self.hukusuu_sigeki_flag:
            if self.toigo_flag:
                rensoubun_numbers = [[0]]
            else:
                rensoubun_numbers = [[0]]

        with open(output_csv, 'w', newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for rensoubun_number in rensoubun_numbers:

                # 連想文番号を書き出し
                writer.writerow(rensoubun_number)

                if another_flag == 293:
                    def ana_ana_hukusuu(hukusuu_sigeki_flag, dict_keywords, kankei, kan=None, rank_p=None):
                        word_num_dicts = []
                        word_num_dict = {}
                        scores = []
                        if hukusuu_sigeki_flag:
                            keywords = dict_keywords

                        for keyword in keywords:
                            if kan == "全部":
                                word_num_dict[keyword] = 0.0
                            else:
                                if self.toigo[keyword] == kan:
                                    word_num_dict[keyword] = 0.0
                                else:
                                    pass
                        # 最終的に割った後の数を全部足すことでスコアとなる.
                        for i, result in enumerate(results.itertuples()):
                            if result[3] in rensoubun_number:

                                # 刺激語を複数表記する場合は、result[2]で判断できる
                                # 解答はresult[5]で判断できる
                                keyword = ast.literal_eval(result[5])[0]
                                human_words_rank = int(result[6])
                                if keyword in word_num_dict.keys():
                                    if eval_flag == "MRR":
                                        if human_words_rank == 0:
                                            # print(keyword, result[3])
                                            word_num_dict[keyword] += 0.0
                                        else:
                                            word_num_dict[keyword] += 1.0 / human_words_rank
                                    elif eval_flag == "p":
                                        if human_words_rank == 0:
                                            pass
                                        elif human_words_rank <= rank_p:
                                            word_num_dict[keyword] += 1.0
                                    else:
                                        pass

                        # 連想文の平均
                        score = sum(word_num_dict.values()) / len(rensoubun_number) / len(word_num_dict.keys())
                        scores.append([score])

                        # 刺激語ごとの平均
                        for key, value in word_num_dict.items():
                            word_num_dict[key] = word_num_dict[key] / len(rensoubun_number)
                        word_num_dicts.append([word_num_dict])

                        writer.writerows([kan, str(p)])
                        writer.writerows(scores)
                        writer.writerows(word_num_dicts)
                        writer.writerow("")

                    kankei_list = ["全部",
                                   "仲間", "部分", "色", "季節", "家の中である場所",
                                   "どんなときに持っていく", "行事", "メニュー",
                                   "使ってすること", "都道府県",
                                   "スポーツ", "場所", "国"]

                    for kan in kankei_list:
                        if eval_flag == "MRR":
                            ana_ana_hukusuu(hukusuu_sigeki_flag=self.hukusuu_sigeki_flag, dict_keywords=self.dict_keywords, kankei=self.kankei, kan=kan)
                        elif eval_flag == "p":
                            for p in ps:
                                ana_ana_hukusuu(hukusuu_sigeki_flag=self.hukusuu_sigeki_flag, dict_keywords=self.dict_keywords, kankei=self.kankei, kan=kan, rank_p=p)



### config ###

# BERTから単語を出力するか(results_csvを作成するか)
rensou_flag = True

# 集計するか(analysisファイルを作成するか)
analysis_flag = True

# 使用するフレームワーク
framework_flag = "tf"

# 使用するBERTモデル
model_flag = "cl-tohoku"

# 連想文のMASKに鍵括弧を付けるか，刺激語に鍵括弧を付けるか
mask_kaxtuko_flag = True

# 出力を名寄せするか(基本的にTrue)
output_nayose_flag = True

# 名詞抽出は何を使うか
extract_verb_flag = "mecab"

# 複数刺激語バージョンかどうか(基本的にTrue)
hukusuu_sigeki_flag = True

# 問い語(=限定語)を使用する場合はTrue
toigo_flag = True

# 刺激語の数(喚語資料を使用する場合は 5 )
sigeki_num = 5

# 評価方法
# pは、上位n語に正解語が含まれているかどうか(言語処理学会2022ではこの評価方法を用いた)
# MRRは、MRRスコアを算出する
eval_flag = "p"

# 上位n語まで集計対象とするか
# ps = [1]
# の場合は 上位1語だけ対象
# ps = [5]
# の場合は 上位5語が対象
ps = [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 150]
# の場合は 上位1語、上位2語、...上位150語が対象

# 使用するデータセット。
# "複数単語からの連想実験/データセット/" の中にある
hukusuu_sigeki_dataset = "喚語資料_除去2"

# BERTから出力する単語の数
max_association = 150
max_word_num = 150

# analysis_anotherの集計方式(複数→1つの場合は 293 でOK)
# 293...複数→1つの連想で行う集計
another_flag = 293

# 分析バージョン
analysis_version = "human5_nayose_disred_output_nayose"

# この処理に共通する保存パス
hiduke = "0420"

# 出力するディレクトリ名を決めるための処理
if mask_kaxtuko_flag:
    kaxtuko_name = "「」有"
else:
    kaxtuko_name = "無"
if another_flag == 293:
    another_name = "hukusuu_sigeki1127"
else:
    another_name = "no_analysis"
if hukusuu_sigeki_flag:
    hukusuu_sigeki_name = "hukusuu_sigeki_%s" % sigeki_num
else:
    hukusuu_sigeki_name = "no_hukusuu_sigeki"
if toigo_flag:
    toigo_name = "toigo"
else:
    toigo_name = "no_toigo"

save_dir_name = (hiduke, max_association, kaxtuko_name, analysis_version, another_name, model_flag, hukusuu_sigeki_name, toigo_name, extract_verb_flag, eval_flag)

save_dir = "result/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s/" % save_dir_name
os.makedirs(save_dir, exist_ok=True)

results_csv = save_dir + "result.csv"
results_csv_attention = save_dir + "result_attentions_and_namas.csv"

output1_csv = save_dir + "analysis.csv"
output2_csv = save_dir + "analysis_analysis.csv"



### 実験 ###

# bert_associationをインスタンス化
bert_association = BertAssociation(framework_flag=framework_flag, model_flag=model_flag, hukusuu_sigeki_flag=hukusuu_sigeki_flag, sigeki_num=sigeki_num, hukusuu_sigeki_dataset=hukusuu_sigeki_dataset, toigo_flag=toigo_flag, extract_verb_flag=extract_verb_flag)

# 単語を出力する
if rensou_flag:
    bert_association(results_csv=results_csv, results_csv_attention=results_csv_attention, max_association=max_association, mask_kaxtuko_flag=mask_kaxtuko_flag, output_nayose_flag=output_nayose_flag)

# 集計する
if analysis_flag:
    if another_flag == 0 or another_flag == 293:
        bert_association.analysis_result_match_nayose(results_csv=results_csv, output_csv=output1_csv, bert_interval=1, max_word_num=max_word_num, another_flag=another_flag)
        bert_association.analysis_analysis_result_match(results_csv=output1_csv, output_csv=output2_csv, another_flag=another_flag, eval_flag=eval_flag, ps=ps)