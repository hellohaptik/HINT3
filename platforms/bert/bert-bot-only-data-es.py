import argparse
import collections
import datetime
import logging
import os
import random
import sys
from logging.handlers import WatchedFileHandler
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch

assert(torch.cuda.is_available())

logger = logging.getLogger(__name__)
now_ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
OOS_CLASS = 'NO_NODES_DETECTED'


def str2bool(v: Any):
    # https://stackoverflow.com/q/15008758/3697191
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def random_seed(seed_value: int, use_cuda: bool):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: str):
    global logger
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.DEBUG)
    handler = WatchedFileHandler(f'{output_dir}/run_logs.log')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)


def log(*parts):
    logger.info(' '.join([str(part) for part in parts]))
    logger.info("=" * 80)


def make_st_args(cmd_args):
    args = [
        ('fp16', False),
        ('output_dir', f'{cmd_args.output_dir}/'),
        ('best_model_dir', f'{cmd_args.output_dir}/best_model/'),
        ('tensorboard_dir', f'{cmd_args.output_dir}/tblogs/'),
        ('manual_seed', cmd_args.seed),
        ('do_lower_case', cmd_args.do_lower_case),
        ('learning_rate', cmd_args.learning_rate),
        ('train_batch_size', cmd_args.batch_size),
        ('eval_batch_size', cmd_args.batch_size),
        ('num_train_epochs', cmd_args.epochs),
        ('gradient_accumulation_steps', 1),
        ('max_seq_length', 512),
        ('overwrite_output_dir', True),
        ('reprocess_input_data', True),
        ('save_best_model', True),
        ('save_eval_checkpoints', False),
        ('save_model_every_epoch', False),
        ('save_optimizer_and_scheduler', True),
        ('save_steps', -1),
        ('evaluate_during_training', True),
        ('evaluate_during_training_silent', False),
        ('evaluate_during_training_steps', cmd_args.eval_every_n_steps),
        ('evaluate_during_training_verbose', True),
    ]

    if cmd_args.use_early_stopping:
        args.append(('use_early_stopping', True))
        args.append(('early_stopping_consider_epochs', True))
        args.append(('early_stopping_metric', 'eval_loss'))
        args.append(('early_stopping_metric_minimize', True))
        args.append(('early_stopping_delta', cmd_args.early_stopping_delta))
        args.append(('early_stopping_patience', cmd_args.early_stopping_patience))
    else:
        args.append(('use_early_stopping', False))

    return dict(args)


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={'sentence': 'text', 'label': 'labels'})
    df = df[['text', 'labels']]
    return df


def get_labels_map(df):
    labels = sorted(df['labels'].unique().tolist())
    label2id = collections.OrderedDict(zip(labels, range(len(labels))))
    return label2id


def f1_at_threshold(preds, y_true, labels_list, threshold):
    labels_list = labels_list + [OOS_CLASS]
    idxs = np.argmax(preds, axis=1)
    scores = preds[np.arange(preds.shape[0]), idxs]
    idxs[scores < threshold] = -1
    y_pred = [labels_list[i] for i in idxs]
    return f1_score(y_true=y_true, y_pred=y_pred, average='weighted')


def run_experiment(cmd_args):
    setup_logging(cmd_args.output_dir)
    log('Run args', vars(cmd_args))
    torch.cuda.empty_cache()
    random_seed(cmd_args.seed, True)
    train_df = read_data(cmd_args.train_file)
    eval_df = read_data(cmd_args.train_file)
    label2id = get_labels_map(train_df)
    test_df = read_data(cmd_args.test_file)
    if cmd_args.eval_frac > 0:
        train_df, eval_df = train_test_split(
            train_df, test_size=cmd_args.eval_frac,
            random_state=cmd_args.seed,
            shuffle=True,
            stratify=train_df['labels']
        )

    log('Train Shape', train_df.shape)
    log('Eval Shape', train_df.shape)
    log('Test Shape', train_df.shape)

    weights = compute_class_weight('balanced', classes=list(label2id.keys()), y=train_df['labels']).tolist()
    log('Class weights', weights)

    args = make_st_args(cmd_args)
    args['labels_list'] = list(label2id.keys())
    args['labels_map'] = label2id

    log('Labels map', label2id)
    log('ST args', args)

    m = ClassificationModel(
        model_type=cmd_args.model_type,
        model_name=cmd_args.model_name,
        num_labels=len(label2id),
        weight=weights,
        args=args)
    m.train_model(train_df=train_df, eval_df=eval_df)
    m = ClassificationModel(
        cmd_args.model_type,
        args['best_model_dir'],
        args=args,
    )
    _, logits = m.predict(test_df['text'])
    preds = softmax(logits, axis=1)
    top_predicted = np.argmax(preds, axis=1)
    
    out_df = test_df.rename(columns={'text': 'sentence', 'labels': 'label'})
    out_df['predicted_node'] = [m.args.labels_list[top_predicted[i]] for i in range(len(test_df))]
    out_df['predicted_node_score'] = [preds[i][top_predicted[i]] for i in range(len(test_df))]
    out_df.to_csv(f'{cmd_args.output_dir}/predictions.csv', columns=['sentence', 'label', 'predicted_node', 'predicted_node_score'], index=False)
    
    test_df['predictions'] = [dict(zip(m.args.labels_list, preds[i])) for i in range(len(test_df))]
    test_df.to_json(f'{cmd_args.output_dir}/predictions.jsonl', orient='records', lines=True)
    for t in range(0, 101, 5):
        t = t / 100.0
        f1 = f1_at_threshold(preds, test_df['labels'], m.args.labels_list, t)
        log(f'F1 @ t={t}', f1)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', required=True, type=str)
    parser.add_argument('--test_file', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--model_type', required=False, default='bert', type=str)
    parser.add_argument('--model_name', required=False, default='bert-base-uncased', type=str)
    parser.add_argument('--do_lower_case', required=False, default=True, type=str2bool)
    parser.add_argument('--seed', required=False, default=42, type=int)
    parser.add_argument('--learning_rate', required=False, default=0.00004, type=float)
    parser.add_argument('--batch_size', required=False, default=16, type=int)
    parser.add_argument('--epochs', required=False, default=10, type=int)
    parser.add_argument('--eval_frac', required=False, default=0.1, type=float)
    parser.add_argument('--eval_every_n_steps', required=False, default=100, type=int)
    parser.add_argument('--use_early_stopping', required=False, default=True, type=str2bool)
    parser.add_argument('--early_stopping_patience', required=False, default=5, type=int)
    parser.add_argument('--early_stopping_delta', required=False, default=0.00005, type=float)
    cmd_args = parser.parse_args()
    cmd_args.output_dir = f'{cmd_args.output_dir.rstrip("/")}' # /{now_ts}'
    os.makedirs(cmd_args.output_dir, exist_ok=True)
    random_seed(cmd_args.seed, True)
    run_experiment(cmd_args)


if __name__ == '__main__':
    main()
