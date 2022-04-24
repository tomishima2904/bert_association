from transformers import BertJapaneseTokenizer, TFBertForMaskedLM, BertConfig
import tensorflow as tf
import fugashi


class Model(object):
    def __init__(self, mecab_option, args) -> None:
        if args.model_opt == 'cl-tohoku':
            # 東北大学の乾研究室が作成した日本語BERTモデル
            model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
            self.model = TFBertForMaskedLM.from_pretrained(model_name, from_pt=True)
            # Mecab
            self.mecab = fugashi.GenericTagger(mecab_option)

        elif args.model_opt == "addparts":
            # 東北大学の乾研究室が作成した日本語BERTモデルを元に、名詞だけを事前学習したBERT
            config = BertConfig.from_json_file('addparts/config.json')
            self.tokenizer = BertJapaneseTokenizer.from_pretrained('addparts/vocab.txt', do_lower_case=False, word_tokenizer_type="mecab", mecab_dic_type='unidic_lite', unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]')
            self.model = TFBertForMaskedLM.from_pretrained('addparts/pytorch_model.bin', config=config, from_pt=True)
            # Mecab
            self.mecab = fugashi.GenericTagger(mecab_option)

        elif self.model_opt == "gmlp":
            from official.nlp.modeling.networks import gmlp_encoder_remake
            from official.nlp.modeling import models
            from official.modeling import tf_utils
            from official.nlp.gmlp import configs

            transformer_encoder = gmlp_encoder_remake.GmlpEncoder(
                vocab_size=32768,
                hidden_size=768,
                num_layers=24,
                sequence_length=128,
                activation=tf.nn.swish,
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                return_all_encoder_outputs=False,
                output_range=None,
                embedding_width=None,
                embedding_layer=None,
                max_sequence_length=None,
                type_vocab_size=0,
                num_attention_heads=0,
                intermediate_size=0,
                dropout_rate=0.0,
                attention_dropout_rate=0.0)
            self.model = models.GmlpPretrainer(
                network=transformer_encoder,
                embedding_table=transformer_encoder.get_embedding_table(),
                num_classes=2,  # The next sentence prediction label has two classes.
                activation=tf_utils.get_activation(
                    configs.GmlpConfig.from_json_file("gMLP/config.json").hidden_act),
                num_token_predictions=80,
                output='logits')

            checkpoint = tf.train.Checkpoint(model=self.model)
            checkpoint.restore("gMLP/gmlp_model_step_2000000.ckpt-5")

            self.tokenizer = BertJapaneseTokenizer.from_pretrained('gMLP/vocab.txt', do_lower_case=False,
                                                                    word_tokenizer_type="mecab",
                                                                    mecab_dic_type='unidic_lite', unk_token='[UNK]',
                                                                    sep_token='[SEP]', pad_token='[PAD]',
                                                                    cls_token='[CLS]', mask_token='[MASK]')
            self.mecab = fugashi.GenericTagger(mecab_option)