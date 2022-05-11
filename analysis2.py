import csv
import ast
import os, sys
import pandas as pd
import numpy as np
from IPython.display import display, HTML

sys.path.append('.')
from file_handler import *
from analysis import Analyzer
from config import args


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

    analysis = Analyzer2(args)
    results_dir = dir_name_getter(args, get_date=args.get_date)
    print(results_dir)
    analysis.attention_visualizasion(results_dir)
