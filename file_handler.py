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
  results = df[['sid', 'stims', 'input_sentence', 'answer', 'output_words', 'output_scores']]
  # [sid, keyword, input_sentence, answer, association_words, association_score]
  # 1...キーワード,
  # 2...入力文の番号,
  # 3...入力文,
  # 4...目的の色(人間の答え?),
  # 5...出力された単語(150単語, 順位順, 出現しなかった場合は-1)
  # 6...出力された単語のスコア
  return results


# analysis_result_matchで使用する書き込み関数
def csv_writer(header:list, result:list, csv_file_path:str):
  with open(csv_file_path, 'w', newline="",  encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      writer.writerows(result)