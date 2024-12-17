#!/bin/bash -e


python planparse/norbert/train_seq_norbert.py --config configs/nb_bert_base.json --epochs 2
python planparse/norbert/train_seq_norbert.py --config configs/nb_bert_large.json --epochs 2
python planparse/norbert/train_seq_norbert.py --config configs/norbert3_extrasmall.json --epochs 2
python planparse/norbert/train_seq_norbert.py --config configs/norbert3_small.json --epochs 2
python planparse/norbert/train_seq_norbert.py --config configs/norbert3_base.json --epochs 2
python planparse/norbert/train_seq_norbert.py --config configs/norbert3_large.json --epochs 2

python planparse/norbert/train_seq_norbert.py --config configs/nb_bert_base.json --epochs 4
python planparse/norbert/train_seq_norbert.py --config configs/nb_bert_large.json --epochs 4
python planparse/norbert/train_seq_norbert.py --config configs/norbert3_extrasmall.json --epochs 4
python planparse/norbert/train_seq_norbert.py --config configs/norbert3_small.json --epochs 4
python planparse/norbert/train_seq_norbert.py --config configs/norbert3_base.json --epochs 4
python planparse/norbert/train_seq_norbert.py --config configs/norbert3_large.json --epochs 4


# python planparse/nb_llama/train_text_gen_seq_cls.py --config configs/nb_llama_1b.json
# python planparse/nb_llama/train_text_gen_seq_cls.py --config configs/nb_llama_3b.json
# python planparse/nb_llama/train_text_gen_seq_cls.py --config configs/norwai_mistral_7b.json
