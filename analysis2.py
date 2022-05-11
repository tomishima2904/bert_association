import csv
import ast
import os, sys
import pandas as pd
import argparse
import numpy as np
from distutils.util import strtobool
from IPython.display import display, HTML

sys.path.append('.')
from file_handler import *
from analysis import Analyzer


class Analyzer2(Analyzer):
    def __init__(self, args) -> None:
        super().__init__(args)


    def _highlight(self, word, attn):
        html_color = f'#{255:02X}{int(255*(1 - attn)):02X}{int(255*(1 - attn)):02X}'
        return f'<span style="background-color: {html_color}">{word}</span>'


    def _mk_html(self, sentence, attns):
        html = ""
        for word, attn in zip(sentence, attns):
            html += ' ' + self._highlight(word, attn)
        return html


    def attention_weights_handler(self, results):
        # tokenized_sentences = results.tokenized_sentence.to_list()
        all_heads_attns = results.attn_weights_of_mask
        converted_attns = []
        if self.args.target_heads == None:
            for attns in all_heads_attns:
                attns = np.array(ast.literal_eval(attns))
                if self.args.avg_flag:
                    averaged_attn = np.average(attns, axis=0)
                    averaged_attn = averaged_attn[np.newaxis, :]
                    converted_attns.append(averaged_attn)
                else:
                    converted_attns.append(attns)

        else: 
            for attns in all_heads_attns:
                attns = np.array(ast.literal_eval(attns))
                bool_list = [head in self.args.target_heads for head in range(len(attns[0]))]
                target_attns = attns[bool_list]
                if self.args.avg_flag:
                    averaged_attn = np.average(target_attns, axis=0)
                    averaged_attn = averaged_attn[np.newaxis, :]
                    converted_attns.append(averaged_attn)
                else:
                    converted_attns(target_attns)

        return converted_attns


    def attention_visualizasion(self, results_csv:str):
        results_path = f'{results_csv}/result_attentions_and_raws.csv'
        results = csv_results_reader(results_path)
        attnetion_weights = self.attention_weights_handler(results)
        tokenized_sentences = results.tokenized_sentence
        tokenized_sentences = [ast.literal_eval(sentence) for sentence in tokenized_sentences]
        assert len(attnetion_weights) == len(results.tokenized_sentence)    
        for sentence, attns in zip(tokenized_sentences, attnetion_weights):
            assert len(sentence) == len(attns[0])
            for attn in attns:
                display(HTML(self._mk_html(sentence, attn)))



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
    parser.add_argument('--packages_path', default='/home/tomishima2904/.local/lib/python3.6/site-packages', type=str, help='path where packages exist')
    parser.add_argument('--dict_mecab', default='ipadic', type=str, help='[unidic_lite, unidic, ipadic]')

    parser.add_argument('--get_date', default=None, help='date_time for hits@k')
    parser.add_argument('--avg_flag', default=True, type=strtobool, help='averaging attn weights or not')
    parser.add_argument('--target_heads', default=None, help='Specify which attention heads to get')
    args = parser.parse_args()

    analysis = Analyzer2(args)
    results_dir = dir_name_getter(args, get_date=args.get_date)
    print(results_dir)
    analysis.attention_visualizasion(results_dir)
