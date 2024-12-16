#!/bin/bash -e


# python planparse/norbert/train_seq_norbert.py --config configs/nb_bert_large.json
# python planparse/norbert/train_seq_norbert.py --config configs/norbert3_extrasmall.json
# python planparse/norbert/train_seq_norbert.py --config configs/norbert3_large.json
# python planparse/norbert/train_seq_norbert.py --config configs/norbert3_small.json3
# python planparse/norbert/train_seq_norbert.py --config configs/norbert3_base.json

python planparse/nb_llama/train_text_gen_seq_cls.py --config configs/nb_llama_1b.json
python planparse/nb_llama/train_text_gen_seq_cls.py --config configs/nb_llama_3b.json
python planparse/nb_llama/train_text_gen_seq_cls.py --config configs/norwai_mistral_7b.json
