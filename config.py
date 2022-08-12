import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser("Setting configurations")

parser.add_argument('--max_words', default=150, type=int, help='num of words from BERT')
parser.add_argument('--brackets_flag', default=True, type=strtobool, help='Adding brackets or not')
parser.add_argument('--model_opt', default='cl-tohoku', type=str, help='Specify a BERT model')
parser.add_argument('--dict_mecab', default='ipadic', type=str, help='[unidic_lite, unidic, ipadic]')
parser.add_argument('--extract_noun_opt', default='mecab', type=str, help='[mecab, ginza]')
parser.add_argument('--category_opt', default='cat', type=str, help='[cat, WOcat, kotoba]')
parser.add_argument('--reverse_flag', default=False, type=strtobool, help='Associate category if true')
# above args considering directory name

parser.add_argument('--analysis_flag', default=True, type=strtobool, help='Analyze or not')
parser.add_argument('--another_analysis', default=293, type=int, help='Specify another method of analysis')
parser.add_argument('--avg_flag', default=True, type=strtobool, help='averaging attn weights or not')
parser.add_argument('--dataset_csv', default='for_fit2022.csv', type=str, help='csv name of dataset')
parser.add_argument('--dataset_dir', default='extract_keywordslist', type=str, help='directory of dataset')
parser.add_argument('--eval_opt', default='p', type=str, help='[p, MRR]')
parser.add_argument('--framework_opt', default='tf', type=str, help='Specify a framework')
parser.add_argument('--get_date', default=None, help='date_time for hits@k')
parser.add_argument('--output_nayose_flag', default=True, type=strtobool, help='Nayose or not')
parser.add_argument('--output_words_from_bert', default=True, type=strtobool, help='Output words from BERT or not')
parser.add_argument('--packages_path', default='/home/tomishima2904/.local/lib/python3.6/site-packages', type=str, help='path where packages exist')
parser.add_argument('--ps', nargs='*', default=[1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 150], type=int, help='Specify ranks for analysis')
parser.add_argument('--sep_flag', default=False, type=strtobool, help='visualizing attention weights with SEP token if True')
parser.add_argument('--summary_num', default=10, type=int, help='number of extract from all output words')
parser.add_argument('--target_heads', nargs='*', default=[-1], type=int, help='Specify which attention heads to get')
parser.add_argument('--target_layer', default=-1, type=int, help='Specify output layer of transformer')

args = parser.parse_args()