import csv
import ast
import os
import pandas as pd
import datetime


# results_csvを読み込む関数
def csv_results_reader(results_csv):
  try:
      df = pd.read_csv(results_csv, header=0, encoding="utf-8", engine="python")
  except:
      df = pd.read_csv(results_csv, header=0, engine="python")
  # print(df)
  return df


# analysis_result_matchで使用する書き込み関数
def csv_writer(header:list, result:list, csv_file_path:str):
  with open(csv_file_path, 'w', newline="",  encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      writer.writerows(result)


def dir_name_getter(args):
    # この処理に共通する保存パス
    if type(args.get_date) is str:
        date_time = args.get_date

    else:
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        date_time = now.strftime('%y%m%d_%H%M%S')

    # 出力するディレクトリ名を決めるための処理

    if args.another_analysis == 293: another_name = "anl"
    else: another_name = "WOanl"

    if args.multi_stims_flag: stims_name = "stims{}".format(args.num_stims)
    else: stims_name = "WOstims"

    save_dir = f"results/{date_time}_{args.max_words}_{args.model_opt}_{args.dict_mecab}_{args.extract_noun_opt}_{file_name_getter(args)}"

    return save_dir


def file_name_getter(args):
    if args.brackets_flag: brkt_flag = 'brkt'
    else: brkt_flag = 'WObrkt'
    if args.reverse_flag: return f'rev_{brkt_flag}'
    else: return f'{args.category_opt}_{brkt_flag}'


def html_writer(body, result_dir:str, output_file:str, args):
    result = f'''
            <html>
            <head>
            <meta charset="utf-8">
            <title>{result_dir}</title>
            </head>
            <body>
            <h2>{result_dir}</h2>
            {body}
            </body>
            </html>
        '''

    save_file_name = f'{result_dir}/{output_file}_{file_name_getter(args)}.html'

    with open(save_file_name, 'w', encoding='utf-8') as f:
        f.write(result)