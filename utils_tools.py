# トークン化された出力を 変換する関数
# masked_indexはリストで返す
def transform_tokenized_text_mecab_tohoku(tokenized_text):
    masked_index = []
    for i, tokenized in enumerate(tokenized_text):
        if tokenized == '*':
            tokenized_text[i] = '[MASK]'
            masked_index.append(i + 1)
            # 「。」をそのまま[SEP]にする処理が原論文と同じになる.
            # *で処理しているせいで, *の後に##が付いてしまっている. 修正.
        if tokenized == '。' or tokenized == '##。':
            tokenized_text[i] = '[SEP]'
        if tokenized == '##」':
            tokenized_text[i] = '」'
        if tokenized == '##*':
            tokenized_text[i] = '[MASK]'
            masked_index.append(i + 1)
        if tokenized == '##、':
            tokenized_text[i] = '、'
    tokenized_text.insert(0, '[CLS]')
    return tokenized_text, masked_index


# 複数→1つの連想：限定語なし
hukusuu_sigeki_sentences = ["{stims}", "から連想する", "言葉", "は", "*", "です。"]

# 複数→1つの連想：限定語なし、鍵括弧「」有り
hukusuu_sigeki_sentences_mask = ["{stims}", "から連想する", "言葉", "は", "「*」", "です。"]


# 複数→1つの連想：限定語あり
hukusuu_sigeki_sentences_toigo = {
        "仲間": {'sentence': ["{stims}", "は*の仲間です。"], 'synonyms': ['範疇', '一部', '集団', 'グループ']},
        "部分": {'sentence': ["{stims}", "からできているものは*です。"], 'synonyms': []}, 
        "色": {'sentence': ["{stims}", "から連想する色は*です。"], 'synonyms': []},
        "季節": {'sentence': ["{stims}", "から連想する季節は*です。"], 'synonyms': []},
        "家の中である場所": {'sentence': ["家の中で", "{stims}", "がある場所は*です。"], 'synonyms': []},
        "どんなときに持っていく": {'sentence': ["{stims}", "は*のときに持っていきます。"], 'synonyms': []},
        "行事": {'sentence': ["{stims}", "から連想される行事は*です。"], 'synonyms': []},
        "メニュー": {'sentence': ["{stims}", "から連想されるメニューは*です。"], 'synonyms': []},
        "使ってすること": {'sentence': ["{stims}", "を使ってすることは*です。"], 'synonyms': []},
        "どのような生き物": {'sentence': ["{stims}", "は*の生き物です。"], 'synonyms': []},
        "都道府県": {'sentence': ["{stims}", "から連想する都道府県は*です。"], 'synonyms': []},
        "スポーツ": {'sentence': ["{stims}", "から想像できるスポーツは*です。"], 'synonyms': []},
        "場所": {'sentence': ["{stims}", "から連想する場所は*です。"], 'synonyms': []},
        "国": {'sentence': ["{stims}", "から連想する国は*です。"], 'synonyms': []},
        "どこ": {'sentence': ["{stims}", "から連想するのは*です。"], 'synonyms': []}    
}

# 複数→1つの連想：限定語あり、鍵括弧「」無し
hukusuu_sigeki_sentences_toigo_mask = {
        "仲間": {'sentence': ["{stims}", "は「*」の仲間です。"], 'synonyms': ['範疇', '一部', '集団', 'グループ']},
        "部分": {'sentence': ["{stims}", "からできているものは「*」です。"], 'synonyms': []},
        "色": {'sentence': ["{stims}", "から連想する色は「*」です。"], 'synonyms': []},
        "季節": {'sentence': ["{stims}", "から連想する季節は「*」です。"], 'synonyms': []},
        "家の中である場所": {'sentence': ["家の中で", "{stims}", "がある場所は「*」です。"], 'synonyms': []},
        "どんなときに持っていく": {'sentence': ["{stims}", "は「*」のときに持っていきます。"], 'synonyms': []},
        "行事": {'sentence': ["{stims}", "から連想される行事は「*」です。"], 'synonyms': []},
        "メニュー": {'sentence': ["{stims}", "から連想されるメニューは「*」です。"], 'synonyms': []},
        "使ってすること": {'sentence': ["{stims}", "を使ってすることは「*」です。"], 'synonyms': []},
        "どのような生き物": {'sentence': ["{stims}", "は「*」のような生き物です。"], 'synonyms': []},
        "都道府県": {'sentence': ["{stims}", "から連想する都道府県は「*」です。"], 'synonyms': []},
        "スポーツ": {'sentence': ["{stims}", "から想像できるスポーツは「*」です。"], 'synonyms': []},
        "場所": {'sentence': ["{stims}", "から連想する場所は「*」です。"], 'synonyms': []},
        "国": {'sentence': ["{stims}", "から連想する国は「*」です。"], 'synonyms': []},
        "どこ": {'sentence': ["{stims}", "から連想するのは「*」です。"], 'synonyms': []}
}