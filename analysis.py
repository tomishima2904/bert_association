from cmath import nan
import csv
import ast
import os, sys
import pandas as pd

sys.path.append('.')
from file_handler import *
from dict_maker import SearchBasedOnStimulations
from config import args
import utils_tools


class Analyzer(object):
    def __init__(self, args) -> None:
        self.args = args

        if args.multi_stims_flag:
            self.hukusuu_sigeki = SearchBasedOnStimulations(args=args)
            self.paraphrase = self.hukusuu_sigeki.get_paraphrase_hukusuu_sigeki()
            self.nayose = self.hukusuu_sigeki.get_nayose_hukusuu_sigeki()
            # 複数の刺激語バージョンにおける、正解と不正解のリスト
            self.dict_keywords = self.hukusuu_sigeki.get_dict()
            # wikipediaの出現頻度とか共起頻度とか分かる（予定）
            # self.toigo = self.hukusuu_sigeki.get_toigo()
            self.kankei = self.hukusuu_sigeki.get_kankei()
            if args.reverse_flag:
                self.categories_and_sentences = utils_tools.base_sentences_and_synonyms

        else:
            pass

    # 出力結果の分析
    def analysis_result_match_nayose(self, results_csv, output_csv, bert_interval=1):
        results = csv_results_reader(results_csv)

        # 出力した単語と正解の連想語が一致する場合にリストにぶち込んだりカウントを足したりする
        result_match = []
        for i, result in enumerate(results.itertuples()):
            if self.args.another_analysis == 293:
                if self.args.reverse_flag:  # カテゴリーを答えさせる場合
                    # human_words = self.categories_and_sentences[result.category]['synonyms']  # 類似語も正解とする場合
                    human_words = [result.category]  # 類似語は正解とせず、代表カテゴリー語のみを正解とする場合
                else:  # 正解語を答えさせる場合
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

                # result_tmp = result.sid, result.stims, result.input_sentence, result.answer, result.category, result.category_synonyms, human_words_rank, bert_and_human_word, bert_and_human_score, not_bert_and_human_word, not_bert_and_human_score, not_bert_and_human_num
                result_tmp = result.sid, result.stims, result.input_sentence, result.answer, result.category, result.category_synonyms, human_words_rank, bert_and_human_word, bert_and_human_score, result.output_words, result.output_scores
                result_match.append(result_tmp)

        # header_results = ['sid', 'stims', 'input_sentence', 'answer', 'category', 'category_synonyms', 'ranks', 'corr_word', 'corr_score', 'err_words', 'err_scores', 'num_err_per_iv']
        header_results = ['sid', 'stims', 'input_sentence', 'answer', 'category', 'category_synonyms', 'ranks', 'corr_word', 'corr_score', 'output_words', 'output_scores']

        csv_writer(header=header_results, result=result_match, csv_file_path=output_csv)


    # スコアを算出する  # Not working!!!!
    def analysis_analysis_result_match(self, results_csv, output_csv):
        results = pd.read_csv(results_csv, header=0, engine="python")

        # 連想文の組み合わせ
        if self.args.multi_stims_flag:
            if self.args.category_opt:
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
                                human_words_rank = int(result.ranks)
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

        results = pd.read_csv(f"{results_dir}/analysis_{file_name_getter(self.args)}.csv", header=0, engine="python", encoding='utf-8')
        extract_results = results[results['sid'].str.contains('-00')]  # 類似語の数は考慮せず、代表カテゴリー語の文だけ抽出
        total_sentences = len(extract_results)  # 80題になるはず(fit2022.csvを使えば)
        total_sentences_WO_nakama = total_sentences - sum(extract_results.category == '仲間')  # 「仲間」を除去したバージョン
        header = ['k', f'hits@k({total_sentences})', 'hits_num', 'hits@k_WO_nakama']
        all_k_resutlts = []
        category_list = list(set(results.category.tolist()))
        category_list.sort()
        for i, at_most_k in enumerate(target_ranks):  # 全体のhits@kを算出する
            masked_rank_at_k = [result_row.ranks <= at_most_k and result_row.ranks != 0 for result_row in extract_results.itertuples()]
            masked_rank_at_k_WO_nakama = [result_row.ranks <= at_most_k and result_row.ranks != 0 and result_row.category != '仲間' for result_row in extract_results.itertuples()]
            total_num_within_k = sum(masked_rank_at_k)
            total_num_within_k_WO_nakama = sum(masked_rank_at_k_WO_nakama)
            hits_ratio = total_num_within_k/total_sentences
            hits_ratio_WO_nakama = total_num_within_k_WO_nakama / total_sentences_WO_nakama
            hits_k_results = [at_most_k, hits_ratio, total_num_within_k, hits_ratio_WO_nakama]

            if self.args.category_opt=='cat':  # カテゴリーごとのhits@kを算出する(類似語別)
                for category in category_list:
                    specifical_df = results[results['category'] == category]
                    specifical_synonyms_list = list(set(specifical_df.category_synonyms.tolist()))
                    specifical_synonyms_list.sort()
                    for synonym in specifical_synonyms_list:
                        spe_syn_df = specifical_df[specifical_df['category_synonyms'] == synonym]
                        spe_syn_total_sentences = len(spe_syn_df)
                        print(category, synonym, spe_syn_total_sentences)
                        spe_syn_masked_rank_at_k = [spe_row.ranks <= at_most_k and spe_row.ranks != 0 for spe_row in spe_syn_df.itertuples()]
                        spe_syn_tatal_num_within_k = sum(spe_syn_masked_rank_at_k)
                        spe_syn_hits_ratio = spe_syn_tatal_num_within_k/spe_syn_total_sentences
                        hits_k_results.append(spe_syn_hits_ratio)

                        if i == 0:
                            spe_header = f'{category}:{synonym}({spe_syn_total_sentences})'
                            header.append(spe_header)

            else:  # カテゴリーごとのhits@kを算出する
                for category in category_list:
                    specifical_df = results[results['category'] == category]
                    specifical_total_sentences = len(specifical_df)
                    specifical_masked_rank_at_k = [spe_row.ranks <= at_most_k and spe_row.ranks != 0 for spe_row in specifical_df.itertuples()]
                    specifical_total_num_within_k = sum(specifical_masked_rank_at_k)
                    specifical_hits_ratio = specifical_total_num_within_k/specifical_total_sentences
                    hits_k_results.append(specifical_hits_ratio)

                    if i == 0:
                        spe_header = f'{category}({specifical_total_sentences})'
                        header.append(spe_header)

            all_k_resutlts.append(hits_k_results)

        print("Done hits at k\n")
        output_file = f"{results_dir}/hits_at_k_{file_name_getter(self.args)}.csv"
        csv_writer(header=header, result=all_k_resutlts, csv_file_path=output_file)



### 実験 ###

if __name__ == '__main__':

    analysis = Analyzer(args)
    results_dir = dir_name_getter(args)
    result_csv = results_dir + f'/result_{file_name_getter(args)}.csv'
    analysis_csv = results_dir + f"/analysis_{file_name_getter(args)}.csv"
    analysis.analysis_result_match_nayose(result_csv, analysis_csv)
    analysis.hits_at_k(results_dir=results_dir, target_ranks=args.ps)