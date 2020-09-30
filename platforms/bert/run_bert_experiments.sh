#!/bin/bash

pip install -U -r up-requirements.txt
#declare -a datasets=("sofmattress" "powerplay11")
declare -a datasets=("curekart")

for dataset in "${datasets[@]}"
do

    python bert-bot-only-data-es.py \
	--train_file  "../train/${dataset}_train.csv" \
	--test_file "../test/${dataset}_test.csv" \
	--output_dir "../bert_models/${dataset}/" \
	--model_type "bert" \
	--model_name  "bert-base-uncased" \
	--do_lower_case true \
	--seed 42 \
	--learning_rate 0.00004 \
	--batch_size 8 \
	--epochs 50 \
	--eval_frac 0.0 \
	--eval_every_n_steps 100 \
	--use_early_stopping true \
	--early_stopping_patience 5 \
	--early_stopping_delta 0.0005 \

    cp "../bert_models/${dataset}/predictions.csv" "../preds/bert_${dataset}.csv"
done


for dataset in "${datasets[@]}"
do

    python bert-bot-only-data-es.py \
	--train_file  "../train/${dataset}_subset_train.csv" \
	--test_file "../test/${dataset}_test.csv" \
	--output_dir "../bert_models/${dataset}_subset/" \
	--model_type "bert" \
	--model_name  "bert-base-uncased" \
	--do_lower_case true \
	--seed 42 \
	--learning_rate 0.00004 \
	--batch_size 8 \
	--epochs 50 \
	--eval_frac 0.0 \
	--eval_every_n_steps 100 \
	--use_early_stopping true \
	--early_stopping_patience 5 \
	--early_stopping_delta 0.0005 \

    cp "../bert_models/${dataset}_subset/predictions.csv" "../preds/bert_${dataset}_subset.csv"
done

pip install -U -r down-requirements.txt
