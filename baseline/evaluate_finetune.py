from __future__ import absolute_import
import os
import sys
import math
import json
import pickle
import random
import logging
import argparse
import statistics
from pathlib import Path
from copy import deepcopy

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import (DataLoader, Dataset, SequentialSampler,
                              RandomSampler, TensorDataset)

import transformers
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaTokenizer, RobertaForMaskedLM)

from transform_dataset import BaseInputExample

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECTS = ['android', 'gwt', 'hibernate-orm', 'jdk', 'joda-time', 'xstream']
JAVA_KEYWORDS = ['public', 'private', 'Public', 'Private', 'final', 'Final', 'new', 'New', 'void',
                 'Void', 'Static', 'static', 'protected', 'Protected', 'extends']
MIN_SPAN, MAX_SPAN = 2, 51


ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def set_seed(seed=42):
    '''Set seed to the same value across system, Python, Numpy, and PyTorch.

    Arguments:
        seed (int): Seed value.
    '''
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_examples(path_to_data, stage):
    filename = f"examples_base_{stage}_ft.pkl"
    path_to_data = Path(path_to_data) / filename
    with open(path_to_data, 'rb') as fileobj:
        examples = pickle.load(fileobj)
    return examples


def construct_dataset_eval(args, examples, tokenizer):
    all_input_ids, all_input_masks, all_labels, all_true_labels = [], [], [], []
    cls_id = tokenizer._convert_token_to_id(tokenizer.cls_token)
    mask_id = tokenizer._convert_token_to_id(tokenizer.mask_token)
    pad_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
    sep_id = tokenizer._convert_token_to_id(tokenizer.sep_token)

    for ex_index, example in tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")

        for glob_example in example.glob_lines:
            code = glob_example["globCode"]
            blank = glob_example["blank"]

            pre_blank, post_blank = code.split("<blank>")
            pre_blank_ids = tokenizer(pre_blank)["input_ids"]
            original_blank_ids = tokenizer(blank)["input_ids"]
            post_blank_ids = tokenizer(post_blank)["input_ids"]

            if len(pre_blank_ids) > args.max_seq_length - 2:
                continue

            example_input_ids, example_input_masks, example_labels = [], [], []
            for num_masks in range(MIN_SPAN, MAX_SPAN + 2):
                blank_padding_length = num_masks - len(original_blank_ids)
                if len(original_blank_ids) > num_masks:
                    blank_ids = original_blank_ids[:num_masks]
                else:
                    blank_ids = original_blank_ids + [pad_id for _ in range(blank_padding_length)]

                input_ids = pre_blank_ids + \
                            [mask_id for _ in range(num_masks)] + \
                            post_blank_ids
                labels = [-100 for _ in range(len(pre_blank_ids))] + \
                         blank_ids + \
                         [-100 for _ in range(len(post_blank_ids))]
                masks = [1 for _ in range(len(input_ids))]

                input_ids = input_ids[:args.max_seq_length - 2]
                labels = labels[:args.max_seq_length - 2]
                masks = masks[:args.max_seq_length - 2]

                padding_length = args.max_seq_length - len(input_ids) - 2
                input_ids = [cls_id] + input_ids + [sep_id] + \
                            [pad_id for _ in range(padding_length)]
                labels = [-100] + labels + [-100] + \
                         [-100 for _ in range(padding_length)]
                masks = [1] + masks + [0] + \
                        [0 for _ in range(padding_length)]

                assert len(input_ids) == args.max_seq_length
                assert len(labels) == args.max_seq_length
                assert len(masks) == args.max_seq_length

                example_input_ids.append(input_ids)
                example_labels.append(labels)
                example_input_masks.append(masks)

            all_input_ids.append(example_input_ids)
            all_input_masks.append(example_input_masks)
            all_labels.append(example_labels)

            original_blank_padding_length = MAX_SPAN - len(original_blank_ids)
            original_blank_ids += [-1 for _ in range(original_blank_padding_length)]
            all_true_labels.append(original_blank_ids)
    dataset = TensorDataset(
            torch.tensor(all_input_ids, dtype=torch.long),
            torch.tensor(all_input_masks, dtype=torch.long),
            torch.tensor(all_labels, dtype=torch.long),
            torch.tensor(all_true_labels, dtype=torch.long)
    )
    return dataset


def make_dataloader(args, examples, stage):
    batch_size = args.eval_batch_size
    all_examples = deepcopy(examples)
    project_examples = dict(zip(PROJECTS, [[] for _ in PROJECTS]))

    # Filter by projects
    for example in examples:
        for project in PROJECTS:
            if project in example.eid:
                project_examples[project].append(example)

    assert len(examples) == sum([len(x) for x in list(project_examples.values())])

    datasets = [construct_dataset_eval(args, all_examples, tokenizer)] + \
              [construct_dataset_eval(args, item, tokenizer) for item in project_examples.values()]

    dataloaders = [DataLoader(dataset, sampler=SequentialSampler(dataset),
                              batch_size=batch_size) for dataset in datasets]

    return dataloaders


def evaluate(args, model, dataloader, tokenizer):
    '''Computes evaluation metrics for the examples in data loader.

    Arguments:
        args (argparse.Arguments):
        model (torch.nn.Model): Trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): Data loader object.

    Returns:
        (dict): Evaluation loss and Accuracy. 
    '''
    cls_id = tokenizer._convert_token_to_id(tokenizer.cls_token)
    mask_id = tokenizer._convert_token_to_id(tokenizer.mask_token)
    pad_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
    sep_id = tokenizer._convert_token_to_id(tokenizer.sep_token)

    # Tracking variables
    subtoken_preds, subtoken_true = [], []
    word_preds, word_true = [], []

    for batch in tqdm(dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        scores_list = [{'prob': 0, 'preds': None, 'prediction': None} for _ in range(len(batch[0]))]
        num_spans = batch[0].shape[1]

        for i in range(num_spans):
            batch_input_ids = batch[0][:, i].contiguous()
            batch_attention_mask = batch[1][:, i].contiguous()
            batch_labels = batch[2][:, i].contiguous()

            for eid, example_input_ids in enumerate(batch_input_ids):
                try:
                    example_input_ids = example_input_ids.unsqueeze(0)
                    example_mask_idx = [np.argwhere(example_input_ids.cpu().numpy()[0] == mask_id)][0].transpose()[0,:].tolist()
                    with torch.no_grad():
                        output = model(example_input_ids)[0]
                    example_preds = output[:, example_mask_idx, :]
                    example_scores, example_idx = torch.topk(example_preds, k=1)
                    example_scores, example_idx = example_scores.view(-1).detach(), example_idx.view(-1).detach()

                    _pred_toks = tokenizer.convert_ids_to_tokens(example_idx)
                    pred_toks = []
                    for tok in _pred_toks:
                        _tok = tok.replace('Ġ','')
                        if _tok not in JAVA_KEYWORDS:
                            pred_toks.append(_tok)
                    pred = ''.join(pred_toks)

                    avg_prob = sum(example_scores.tolist()) / len(example_scores.tolist())
                    if avg_prob > scores_list[eid]['prob']:
                        scores_list[eid] = {'prob': avg_prob, 'preds': example_preds, 'prediction': pred}
                except:
                    pass

        for eid, example_scores in enumerate(scores_list):
            if scores_list[eid]['prediction'] is not None:
                example_true = batch[3][eid].tolist()
                example_true = [x for x in example_true if x != -1]
                example_subtoken_true = tokenizer.convert_ids_to_tokens(example_true)

                example_prediction = scores_list[eid]['prediction']
                example_subtoken_preds = tokenizer.tokenize(example_prediction)

                subtoken_preds.append(example_subtoken_preds)
                subtoken_true.append(example_subtoken_true)
                word_preds.append(example_prediction)
                word_true.append(''.join(example_subtoken_true))

    word_accuracy = sum([1 for x, y in zip(word_true, word_preds) if x == y]) / len(word_true)
    bleu_score = corpus_bleu([[x] for x in subtoken_true], [y for y in subtoken_preds], weights=(0.5, 0.5))
    rouge_scores = [ROUGE_SCORER.score(example_pred, word_true[i]) for i, example_pred in enumerate(word_preds)]
    avg_rouge_score = statistics.mean([score['rougeL'].fmeasure for score in rouge_scores])

    print(f'EM Accuracy: {word_accuracy}\tROUGE-L: {avg_rouge_score}\tBLEU-2: {bleu_score}')

    return {
        'word_preds': word_preds,
        'word_true': word_true,
        'EM-Accuracy': word_accuracy,
        'ROUGE-L': avg_rouge_score,
        'BLEU-2': bleu_score,
    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to datasets directory.")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Output directory for model predictions.")

    ## Experiment parameters
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="Maximum sequence length.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--dropout_rate", default=0.2, type=float,
                        help="Dropout rate.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # Print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    logger.warning(f"Device: {args.device}, Number of GPU's: {args.n_gpu}")

    # Set seed
    set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')
    config = RobertaConfig.from_pretrained('microsoft/codebert-base-mlm')

    if args.load_model_path is not None:
        logger.info(f"Reload model from {args.load_model_path}")
        pt_save_key = ''
        model = RobertaForMaskedLM(config=config)
        model.load_state_dict(torch.load(args.load_model_path), strict=False)
    else:
        logger.info(f"Loading off-the-shelf CodeBERT model")
        pt_save_key = 'base'
        model = RobertaForMaskedLM.from_pretrained('microsoft/codebert-base-mlm', config=config)

    model.to(args.device)

    print(model)
    print()
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    # Prepare test data loader.
    logger.info('Loading test data.')
    eval_examples = get_examples(args.data_dir, 'test')
    logger.info('Constructing data loader for test data.')
    eval_dataloaders = make_dataloader(args, eval_examples, 'test')

    # Start testing
    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(eval_examples)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    logger.info('Measuring performance on test set.')

    # Put the model in evaluation mode--the dropout layers behave
    # differently during evaluation.
    model.eval()

    # Record combined-projects evaluation statistics.
    combined_stats = evaluate(args, model, eval_dataloaders[0], tokenizer)
    logger.info(f"Combined EM-Accuracy: {combined_stats['EM-Accuracy']}")

    with open(f'combined_stats_{pt_save_key}_RUN{RUN}_10.pkl', 'wb') as f:
        pickle.dump(combined_stats, f)

    # Record project-wise evaluation statistics.
    for i, eval_dataloader in enumerate(eval_dataloaders[1:]):
        stats = evaluate(args, model, eval_dataloader, tokenizer)
        logger.info(f"{PROJECTS[i].upper()}\t" + \
                    f"EM-Accuracy: {stats['EM-Accuracy']}")

        with open(f"{PROJECTS[i]}_stats_{pt_save_key}_RUN{RUN}_10.pkl", 'wb') as f:
            pickle.dump(stats, f)
