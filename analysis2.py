import csv
import ast
import os, sys
import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn import preprocessing
from IPython.display import display, HTML

sys.path.append('.')
from file_handler import *
from analysis import Analyzer
from config import args


class Analyzer2(Analyzer):
    def __init__(self, args) -> None:
        super().__init__(args)


    # to highlitht a word depending on attn
    def _highlight(self, word, attn):
        html_color = f'#{255:02X}{int(255*(1 - attn)):02X}{int(255*(1 - attn)):02X}'
        return f'<span style="background-color: {html_color}">{word}</span>'


    # make a highlighted sentence jointing highlighted words
    def _mk_html(self, sentence, attns):
        html = ""
        for word, attn in zip(sentence, attns):
            html += ' ' + self._highlight(word, attn)
        return html


    # attneion-weightに対して各ヘッドの平均を取ったりする
    def attention_weights_handler(self, results):
        # tokenized_sentences = results.tokenized_sentence.to_list()
        all_heads_attns = results.attn_weights_of_mask
        converted_attns = []
        # target_headsが-1を含んでいたら全ヘッドを対象とする
        if -1 in self.args.target_heads:
            for attns in all_heads_attns:
                attns = np.array(ast.literal_eval(attns))
                if self.args.avg_flag:  # 各ヘッドの平均をとる
                    averaged_attn = np.average(attns, axis=0)
                    averaged_attn = averaged_attn[np.newaxis, :]
                    converted_attns.append(averaged_attn)
                else:  # 平均を取らずそのまま返す
                    converted_attns.append(attns)

        # 全ヘッドは対象としない場合
        else:
            for attns in all_heads_attns:
                attns = np.array(ast.literal_eval(attns))
                bool_list = [head in self.args.target_heads for head in range(12)]
                target_attns = attns[bool_list]
                if self.args.avg_flag:
                    averaged_attn = np.average(target_attns, axis=0)
                    averaged_attn = averaged_attn[np.newaxis, :]
                    converted_attns.append(averaged_attn)
                else:
                    converted_attns.append(target_attns)

        return converted_attns

    # attention を可視化する
    def attention_visualizasion(self, results_csv:str):
        results_path = f'{results_csv}/result_attentions_and_raws_{file_name_getter(self.args)}.csv'
        results = csv_results_reader(results_path)
        results_path = f'{results_csv}/result_{file_name_getter(self.args)}.csv'  # 対象となるcsvファイルのパス
        summary_results = csv_results_reader(results_path)
        attnetion_weights = self.attention_weights_handler(results)
        tokenized_sentences = results.tokenized_sentence
        tokenized_sentences = [ast.literal_eval(sentence) for sentence in tokenized_sentences]
        assert len(attnetion_weights) == len(results.tokenized_sentence)

        color_bar_attn = [i*0.05 for i in range(21)]  # color bar's attention weight
        color_bar_str = [f'{i:.2f}' for i in color_bar_attn]  # color bar
        color_bar = f' >>>>>{self._mk_html(color_bar_str, color_bar_attn)} <<<<< <br>\n'  # visualize color bar
        displayed_color_bar_indices = [0]
        result_html = color_bar

        for sid, sentence, attns, category, answer, output_words in zip(results.sid, tokenized_sentences, attnetion_weights, results.category, results.answer, summary_results.output_words):
            assert len(sentence) == len(attns[0])
            if not self.args.sep_flag:  # Remove [SEP] token If sep_flag is NOT True
                del sentence[-1]
                attns = [attn[:-1] for attn in attns]
                # attns = softmax(attns, axis=1)  # normalize by softmax
                attns = preprocessing.minmax_scale(attns, axis=1)  # normalize attneion-weight after removing [SEP] token

            if int(sid[:3]) % 10 == 0 and int(sid[:3]) not in displayed_color_bar_indices:  # display a color bar per 10 sentences
                result_html += color_bar
                displayed_color_bar_indices.append(int(sid[:3]))

            # display averaged attention
            if self.args.avg_flag:
                for attn in attns:
                    attn_output_html = f'{sid}: {self._mk_html(sentence, attn)}'
            # display each head's attention
            else:
                if -1 in self.args.target_heads: target_heads = [head for head in range(12)]
                else: target_heads = self.args.target_heads
                for head, attn in zip(target_heads, attns):
                    attn_output_html = f'{sid}: {self._mk_html(sentence, attn)}'

            # display(HTML(attn_output_html))  # uncomment out to display on jupyter
            tagged_attn_output_html = f'{attn_output_html}  {str(ast.literal_eval(output_words)[:5])}<br>\n'
            result_html += tagged_attn_output_html

        if self.args.avg_flag: avg_name = 'avg'
        else: 'raw'
        if self.args.sep_flag: sep_name = 'sep'
        else: sep_name = 'WOsep'
        output_file = f'visu_{avg_name}_{file_name_getter(self.args)}_{sep_name}'
        html_writer(body=result_html, result_dir=results_csv, output_file=output_file, args=self.args)



### 実験 ###

if __name__ == '__main__':

    analysis = Analyzer2(args)
    results_dir = dir_name_getter(args)
    print(results_dir)
    result_csv = results_dir + f'/result_{file_name_getter(args)}.csv'
    analysis_csv = results_dir + f"/analysis_{file_name_getter(args)}.csv"
    analysis.analysis_result_match_nayose(result_csv, analysis_csv)
    analysis.hits_at_k(results_dir=results_dir, target_ranks=args.ps)
    analysis.attention_visualizasion(results_dir)
