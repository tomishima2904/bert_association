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

    
    def attention_weights_handler(self, results):
        # tokenized_sentences = results.tokenized_sentence.to_list()
        all_heads_attns = results.attn_weights_of_mask
        converted_attns = []
        if -1 in self.args.target_heads:
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
                bool_list = [head in self.args.target_heads for head in range(12)]
                target_attns = attns[bool_list]
                if self.args.avg_flag:
                    averaged_attn = np.average(target_attns, axis=0)
                    averaged_attn = averaged_attn[np.newaxis, :]
                    converted_attns.append(averaged_attn)
                else:
                    converted_attns.append(target_attns)

        return converted_attns


    def attention_visualizasion(self, results_csv:str):
        results_path = f'{results_csv}/result_attentions_and_raws.csv'
        results = csv_results_reader(results_path)
        attnetion_weights = self.attention_weights_handler(results)
        tokenized_sentences = results.tokenized_sentence
        tokenized_sentences = [ast.literal_eval(sentence) for sentence in tokenized_sentences]
        assert len(attnetion_weights) == len(results.tokenized_sentence)    

        color_bar_attn = [i*0.05 for i in range(21)]
        color_bar_str = [f'{i:.2f}' for i in color_bar_attn]
        color_bar = f'<p>color bar >>>{self._mk_html(color_bar_str, color_bar_attn)}<<< color bar<p/>\n'
        result_html = color_bar

        for sid, sentence, attns in zip(results.sid, tokenized_sentences, attnetion_weights):
            assert len(sentence) == len(attns[0])
            if sid % 10 == 0 and sid != 0: result_html += color_bar
            if self.args.avg_flag:
                for attn in attns:
                    attn_output_html = f'{sid:3d}:' +self._mk_html(sentence, attn)
                    
            else:
                if self.args.target_heads == None: target_heads = [head for head in range(12)]
                else: target_heads = self.args.target_heads
                for head, attn in zip(target_heads, attns):
                    attn_output_html = f'{sid:3d}-{head:2d}:' + self._mk_html(sentence, attn)

            # display(HTML(attn_output_html))  # uncomment out to display on jupyter
            tagged_attn_output_html = f'<p>{attn_output_html}</p>\n'
            result_html += tagged_attn_output_html            

        if self.args.avg_flag: output_file = 'visu_avg'
        else: output_file = 'visu_raw'
        html_writer(body=result_html, result_dir=results_csv, output_file=output_file)
        


### 実験 ###

if __name__ == '__main__':    

    analysis = Analyzer2(args)
    results_dir = dir_name_getter(args, get_date=args.get_date)
    print(results_dir)
    analysis.attention_visualizasion(results_dir)
