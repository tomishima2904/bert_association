import csv
import ast
import os
import pandas as pd
import argparse
from distutils.util import strtobool

from file_handler import *
from extract_hukusuu import SearchFromHukusuuSigeki


class Analyzer(object):
    def __init__(self, args) -> None:
        self.args = args

        if args.multi_stims_flag:
            self.hukusuu_sigeki = SearchFromHukusuuSigeki(dataset=args.dataset, num_stims=args.num_stims)
            self.paraphrase = self.hukusuu_sigeki.get_paraphrase_hukusuu_sigeki()
            self.nayose = self.hukusuu_sigeki.get_nayose_hukusuu_sigeki()
            # 複数の刺激語バージョンにおける、正解と不正解のリスト
            self.dict_keywords = self.hukusuu_sigeki.get_dict()
            # wikipediaの出現頻度とか共起頻度とか分かる（予定）
            self.toigo = self.hukusuu_sigeki.get_toigo()
            self.kankei = self.hukusuu_sigeki.get_kankei()
        else:
            pass

    # 出力結果の分析
    def analysis_result_match_nayose(self, results_csv, output_csv, bert_interval):
        results = csv_results_reader(results_csv)

        # 出力した単語と正解の連想語が一致する場合にリストにぶち込んだりカウントを足したりする
        result_match = []
        for i, result in enumerate(results.itertuples()):
            if self.args.another_analysis == 293:
                human_words = ast.literal_eval(result.answer)

                for l, human_word in enumerate(human_words):
                    human_words[l] = ''.join([k for k in human_word if not k.isdigit()])

                # 出力された単語とスコアのリスト
                result_str_to_list_words = ast.literal_eval(result.output_words)
                result_str_to_list_score = ast.literal_eval(result.output_scores)

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
                for j in range(self.args.max_words):
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

                result_tmp = i, result.stims, result.input_sentence, result.answer, human_words_rank, bert_and_human_word, bert_and_human_score, not_bert_and_human_word, not_bert_and_human_score, not_bert_and_human_num
                result_match.append(result_tmp)

        header_results = ['sid', 'stims', 'input_sentence', 'answer', 'rank', 'corr_word', 'corr_score', 'err_words', 'err_scores', 'num_err_per_iv']

        csv_writer(header=header_results, result=result_match, csv_file_path=output_csv)


    # スコアを算出する
    def analysis_analysis_result_match(self, results_csv, output_csv):
        results = pd.read_csv(results_csv, header=0, engine="python")      

        # 連想文の組み合わせ
        if self.args.multi_stims_flag:
            if self.args.category_flag:
                rensoubun_numbers = [[0]]
            else:
                rensoubun_numbers = [[0]]

        with open(output_csv, 'w', newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for rensoubun_number in rensoubun_numbers:

                # 連想文番号を書き出し
                writer.writerow(rensoubun_number)

                if self.args.another_analysis == 293:
                    def ana_ana_hukusuu(dict_keywords, kankei, kan=None, rank_p=None):
                        word_num_dicts = []
                        word_num_dict = {}
                        scores = []
                        if self.args.multi_stims_flag:
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
                            if result.sid in rensoubun_number:

                                # 刺激語を複数表記する場合は、result[2]で判断できる
                                # 解答はresult[5]で判断できる
                                keyword = ast.literal_eval(result.stims)[0]
                                human_words_rank = int(result.rank)
                                if keyword in word_num_dict.keys():
                                    if self.args.eval_flag == "MRR":
                                        if human_words_rank == 0:
                                            # print(keyword, result[3])
                                            word_num_dict[keyword] += 0.0
                                        else:
                                            word_num_dict[keyword] += 1.0 / human_words_rank
                                    elif self.args.eval_flag == "p":
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
                        if self.args.eval_opt == "MRR":
                            ana_ana_hukusuu(dict_keywords=self.dict_keywords, kankei=self.kankei, kan=kan)
                        elif self.args.eval_opt == "p":
                            for p in self.args.ps:
                                ana_ana_hukusuu(dict_keywords=self.dict_keywords, kankei=self.kankei, kan=kan, rank_p=p)


    def hits_at_k(self, results_dir, target_ranks:list):
        
        results = pd.read_csv(f"{results_dir}analysis.csv", header=0, engine="python")   
        total_sentences = len(results)
        header = ['k', 'hits@k', 'hits_num']
        all_k_resutlts = []

        for at_most_k in target_ranks:
          masked_rank_at_k = [result_row.rank <= at_most_k and result_row.rank != 0 for result_row in results.itertuples()]
          total_num_within_k = sum(masked_rank_at_k)
          hits_ratio = total_num_within_k/total_sentences
          all_k_resutlts.append([at_most_k, hits_ratio, total_num_within_k])

        print(all_k_resutlts)
        output_file = f"{results_dir}hits_at_k.csv"
        csv_writer(header=header, result=all_k_resutlts, csv_file_path=output_file)
        





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
    parser.add_argument('--mecab_path', default="-d /usr/local/lib/python3.6/site-packages/ipadic/dicdir -r /usr/local/lib/python3.6/site-packages/ipadic/dicdir/mecabrc", type=str, help='path where mecab is')
    parser.add_argument('--get_date', default=None, help='date_time for hits@k')
    args = parser.parse_args()

    analysis = Analyzer(args)
    results_dir = dir_name_getter(args, get_date=args.get_date)
    analysis.hits_at_k(results_dir=results_dir, target_ranks=args.ps)