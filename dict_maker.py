import glob
import pandas as pd
import ast


class SearchFromHukusuuSigeki():
    def __init__(self, args, path_for_jupyter=None, encoding_type="utf-8"):
        if path_for_jupyter:
            path = path_for_jupyter
        else:
            path = "dataset/"
        self.path_keywords = glob.glob(path + args.dataset_dir + "/" + args.dataset_csv)
        self.path_nayose = glob.glob(path + args.dataset_dir + "/nayose.csv")
        self.path_paraphrase = glob.glob(path + args.dataset_dir + "/paraphrase.csv")
        self.results = {}
        self.results_paraphrase = {}
        self.results_nayose = {}
        self.results_toigo = {}
        self.results_kankei = {}

        # キーワードと解答から dict型　を作成する
        for csv in self.path_keywords:
            # engine="python"を指定しないと動かない
            df = pd.read_csv(csv, encoding=encoding_type, engine="python")
            df_idx_list = [i for i in range(len(df))]
            for idx, row in zip(df_idx_list, df.itertuples()):
                v_row = {'answer': row.answer, 'stims': ast.literal_eval(row.stims), 'category': row.category}
                self.results[idx] = v_row
                # self.results_toigo[key] = row[-1]
                # self.results_kankei[key] = row[-1]

        # paraphrase.csv から dict型 を作成する
        for csv in self.path_paraphrase:
            # engine="python"を指定しないと動かない
            df = pd.read_csv(csv, header=None, encoding=encoding_type, engine="python")
            for row in df.itertuples():
                temp_num = len(row)-1
                key = row[1]
                words = []
                for i in range(temp_num):
                    word = row[i+1]
                    if type(word) == str:
                        words.append(word)
                self.results_paraphrase[key] = words

        # nayose.csv から dict型 を作成する
        for csv in self.path_nayose:
            # engine="python"を指定しないと動かない
            df = pd.read_csv(csv, header=None, encoding=encoding_type, engine="python")
            for row in df.itertuples():
                temp_num = len(row)-1
                key = row[1]
                words = []
                for i in range(temp_num):
                    word = row[i+1]
                    if type(word) == str:
                        words.append(word)
                self.results_nayose[key] = words

    def get_dict(self):
        return self.results

    # def get_num(self):
    #    return self.sigeki_num

    def get_paraphrase_hukusuu_sigeki(self):
        return self.results_paraphrase

    def get_nayose_hukusuu_sigeki(self):
        return self.results_nayose

    def get_human_words(self, key):
        if key in self.results.values():
            return self.results.keys()

    def get_keywords(self):
        return self.results.keys()

    # def get_toigo(self):
    #    return self.results_toigo

    def get_kankei(self):
        return self.results_kankei

    def check_notation_fluctuation(self):
        list = []
        for key, values in self.results.items():
            if key in self.results_nayose[key][0]:
                if key in list:
                    print(key)
                list.append(key)
            else:
                print(key)
            for value in values:
                if value in self.results_paraphrase[value][0]:
                    if value in list:
                        print(value)
                    list.append(value)
                else:
                    print(value)
        print(list)