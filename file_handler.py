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
  print(df)  
  return df


# analysis_result_matchで使用する書き込み関数
def csv_writer(header:list, result:list, csv_file_path:str):
  with open(csv_file_path, 'w', newline="",  encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      writer.writerows(result)


def dir_name_getter(args, get_date=None):
    # この処理に共通する保存パス
    if type(get_date) is str:
        date_time = get_date

    else:       
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        date_time = now.strftime('%y%m%d_%H%M%S')

    # 出力するディレクトリ名を決めるための処理
    if args.brackets_flag: brackets = "brkt"
    else: brackets = "WObrkt"

    if args.another_analysis == 293: another_name = "anl"
    else: another_name = "WOanl"

    if args.multi_stims_flag: stims_name = "stims{}".format(args.num_stims)
    else: stims_name = "WOstims"

    if args.category_flag: cat_name = "cat"
    else: cat_name = "WOcat"

    save_dir = f"results/{date_time}_{args.max_words}_{brackets}_{another_name}_{args.model_opt}_{args.dict_mecab}_{stims_name}_{cat_name}_{args.extract_noun_opt}_{args.eval_opt}"      
    return save_dir
