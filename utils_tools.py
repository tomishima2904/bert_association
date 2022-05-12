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
hukusuu_sigeki_sentences = {
    "0": ["%s", "から連想する", "言葉", "は", "*", "です。"]
}

# 複数→1つの連想：限定語なし、鍵括弧「」有り
hukusuu_sigeki_sentences_mask = {
    "0": ["%s", "から連想する", "言葉", "は", "「*」", "です。"]
}

# 複数→1つの連想：限定語あり
hukusuu_sigeki_sentences_toigo = {
        "仲間": {'sentence': ["%s", "は*の仲間です。"], 'synonyms': ['範疇', '一部', '集団', 'グループ']},
        "部分": {'sentence': ["%s", "からできているものは*です。"], 'synonyms': []}, 
        "色": {'sentence': ["%s", "から連想する色は*です。"], 'synonyms': []},
        "季節": {'sentence': ["%s", "から連想する季節は*です。"], 'synonyms': []},
        "家の中である場所": {'sentence': ["家の中で", "%s", "がある場所は*です。"], 'synonyms': []},
        "どんなときに持っていく": {'sentence': ["%s", "は*のときに持っていきます。"], 'synonyms': []},
        "行事": {'sentence': ["%s", "から連想される行事は*です。"], 'synonyms': []},
        "メニュー": {'sentence': ["%s", "から連想されるメニューは*です。"], 'synonyms': []},
        "使ってすること": {'sentence': ["%s", "を使ってすることは*です。"], 'synonyms': []},
        "どのような生き物": {'sentence': ["%s", "は*の生き物です。"], 'synonyms': []},
        "都道府県": {'sentence': ["%s", "から連想する都道府県は*です。"], 'synonyms': []},
        "スポーツ": {'sentence': ["%s", "から想像できるスポーツは*です。"], 'synonyms': []},
        "場所": {'sentence': ["%s", "から連想する場所は*です。"], 'synonyms': []},
        "国": {'sentence': ["%s", "から連想する国は*です。"], 'synonyms': []},
        "どこ": {'sentence': ["%s", "から連想するのは*です。"], 'synonyms': []}    
}

# 複数→1つの連想：限定語あり、鍵括弧「」無し
hukusuu_sigeki_sentences_toigo_mask = {
        "仲間": {'sentence': ["%s", "は「*」の仲間です。"], 'synonyms': ['範疇', '一部', '集団', 'グループ']},
        "部分": {'sentence': ["%s", "からできているものは「*」です。"], 'synonyms': []},
        "色": {'sentence': ["%s", "から連想する色は「*」です。"], 'synonyms': []},
        "季節": {'sentence': ["%s", "から連想する季節は「*」です。"], 'synonyms': []},
        "家の中である場所": {'sentence': ["家の中で", "%s", "がある場所は「*」です。"], 'synonyms': []},
        "どんなときに持っていく": {'sentence': ["%s", "は「*」のときに持っていきます。"], 'synonyms': []},
        "行事": {'sentence': ["%s", "から連想される行事は「*」です。"], 'synonyms': []},
        "メニュー": {'sentence': ["%s", "から連想されるメニューは「*」です。"], 'synonyms': []},
        "使ってすること": {'sentence': ["%s", "を使ってすることは「*」です。"], 'synonyms': []},
        "どのような生き物": {'sentence': ["%s", "は「*」のような生き物です。"], 'synonyms': []},
        "都道府県": {'sentence': ["%s", "から連想する都道府県は「*」です。"], 'synonyms': []},
        "スポーツ": {'sentence': ["%s", "から想像できるスポーツは「*」です。"], 'synonyms': []},
        "場所": {'sentence': ["%s", "から連想する場所は「*」です。"], 'synonyms': []},
        "国": {'sentence': ["%s", "から連想する国は「*」です。"], 'synonyms': []},
        "どこ": {'sentence': ["%s", "から連想するのは「*」です。"], 'synonyms': []}
}