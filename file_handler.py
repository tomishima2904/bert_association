import csv
import ast
import os
import pandas as pd


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