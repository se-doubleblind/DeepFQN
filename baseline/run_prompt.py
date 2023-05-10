from __future__ import absolute_import
import os
import sys
import json
import pickle
import random
import logging
import argparse
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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECTS = ['android', 'gwt', 'hibernate-orm', 'jdk', 'joda-time', 'xstream']
MODEL_CLS_NAME = 'microsoft/codebert-base-mlm'


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


def get_examples(path_to_data, tuning, stage):
    filename = f"examples_base_{stage}_{tuning}.pkl"
    path_to_data = Path(path_to_data) / filename
    with open(path_to_data, 'rb') as fileobj:
        examples = pickle.load(fileobj)
    return examples


def make_dataloader(args, examples, stage):
    def _construct_dataset(args, examples, tokenizer):
        all_input_ids, all_input_masks, all_labels = [], [], []
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
                blank_ids = tokenizer(blank)["input_ids"]
                post_blank_ids = tokenizer(post_blank)["input_ids"]

                if len(pre_blank_ids) > args.max_seq_length - 2:
                    continue

                input_ids = pre_blank_ids + \
                            [mask_id for _ in range(len(blank_ids))] + \
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

                all_input_ids.append(input_ids)
                all_input_masks.append(masks)
                all_labels.append(labels)

        dataset = TensorDataset(
                torch.tensor(all_input_ids, dtype=torch.long),
                torch.tensor(all_input_masks, dtype=torch.long),
                torch.tensor(all_labels, dtype=torch.long),
        )
        return dataset

    if stage == 'train':
        batch_size = args.train_batch_size
        filename = f"dataloader_pt_{stage}.pkl"
        path_to_file = Path(args.train_dir) / filename

        try:
            with open(str(path_to_file), 'rb') as handler:
                dataloader = pickle.load(handler)
        except FileNotFoundError:
            dataset = _construct_dataset(args, examples, tokenizer)
            sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

            with open(str(path_to_file), 'wb') as handler:
                pickle.dump(dataloader, handler)

        return dataloader
    else:
        batch_size = args.eval_batch_size
        try:
            filename = f"dataloader_pt_{stage}.pkl"
            path_to_file = Path(args.train_dir) / filename

            dataloaders = []
            with open(str(path_to_file), 'rb') as handler:
                dataloaders.append(pickle.load(handler))

            for _id, project in enumerate(PROJECTS):
                p_filename = f"dataloader_pt_{stage}_{project}.pkl"
                path_to_project_file = Path(args.train_dir) / p_filename

                with open(str(path_to_project_file), 'rb') as handler:
                    dataloaders.append(pickle.load(handler))

            return dataloaders
        except FileNotFoundError:
            all_examples = deepcopy(examples)
            project_examples = dict(zip(PROJECTS, [[] for _ in PROJECTS]))

            # Filter by projects
            for example in examples:
                for project in PROJECTS:
                    if project in example.eid:
                        project_examples[project].append(example)

            assert len(examples) == sum([len(x) for x in list(project_examples.values())])

            datasets = [_construct_dataset(args, all_examples, tokenizer)] + \
                       [_construct_dataset(args, item, tokenizer) for item in project_examples.values()]

            dataloaders = [DataLoader(dataset, sampler=SequentialSampler(dataset),
                                      batch_size=batch_size) for dataset in datasets]

            filename = f"dataloader_pt_{stage}.pkl"
            path_to_file = Path(args.train_dir) / filename
            with open(str(path_to_file), 'wb') as handler:
                pickle.dump(dataloaders[0], handler)

            for _id, project in enumerate(PROJECTS):
                p_filename = f"dataloader_pt_{stage}_{project}.pkl"
                path_to_project_file = Path(args.train_dir) / p_filename

                with open(str(path_to_project_file), 'wb') as handler:
                    pickle.dump(dataloaders[_id + 1], handler)

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
    mask_id = tokenizer._convert_token_to_id(tokenizer.mask_token)

    # Tracking variables
    total_eval_loss = 0
    subtoken_preds, subtoken_true = [], []
    word_preds, word_true = [], []

    for batch in dataloader:
        batch = tuple(t.to(args.device) for t in batch)

        # Tell PyTorch not to bother with constructing the compute
        # graph during the forward pass, since this is only needed
        # for backpropogation (training).
        with torch.no_grad():
            outputs = model(
                        input_ids=batch[0],
                        attention_mask=batch[1],
                        labels=batch[2],
            )
            # Accumulate the validation loss.
            batch_loss = outputs.loss
            total_eval_loss += batch_loss.item()

        batch_preds = torch.argmax(outputs.logits, dim=2)

        for eid, example_preds in enumerate(batch_preds):
            example_preds = example_preds[batch[0][eid] == mask_id].tolist()
            example_true = batch[2][eid][batch[0][eid] == mask_id].tolist()
            word_preds.append(example_preds)
            word_true.append(example_true)
            subtoken_preds += example_preds
            subtoken_true += example_true

    # Calculate the average loss over all of the batches.
    eval_loss = total_eval_loss / len(dataloader)
    subtoken_accuracy = sum([1 if subtoken_true[i] == subtoken_preds[i] \
                             else 0 for i in range(len(subtoken_true))]) / len(subtoken_true)
    # word_accuracy = sum([1 if word_true[i] == word_preds[i] \
    #                      else 0 for i in range(len(word_true))]) / len(word_true)

    word_accuracy = sum([1 if all(x == y for x, y in zip(word_true[i], word_preds[i])) \
                         else 0 for i in range(len(word_true))]) / len(word_true)

    print(word_preds[-1])
    print(word_true[-1])

    # Record all statistics.
    return {
        'eval_loss': eval_loss,
        'subtoken_accuracy': subtoken_accuracy,
        'word_accuracy': word_accuracy,
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
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to test on given dataset.")
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
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
    parser.add_argument("--num_train_epochs", default=10, type=int,
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

    # Make directory if output_dir does not exist
    if args.train_dir is not None:
        output_dir = Path(args.train_dir) / f"pretrain_{args.learning_rate}"
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_CLS_NAME)
    config = RobertaConfig.from_pretrained(MODEL_CLS_NAME)

    if args.load_model_path is not None:
        logger.info(f"Reload model from {args.load_model_path}")
        model = RobertaForMaskedLM(config=config)
        model.load_state_dict(torch.load(args.load_model_path), strict=False)
    else:
        model = RobertaForMaskedLM.from_pretrained(MODEL_CLS_NAME, config=config)

    model.to(args.device)

    print(model)
    print()
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    if args.do_train:
        # Prepare training data loader.
        logger.info('Loading training data.')
        train_examples = get_examples(args.data_dir, 'pt', 'train')
        logger.info('Constructing training data loader.')
        train_dataloader = make_dataloader(args, train_examples, 'train')

        # Prepare validation data loader.
        logger.info('Loading validation data.')
        eval_examples = get_examples(args.data_dir, 'pt', 'valid')
        logger.info('Constructing validation data loader.')
        eval_dataloaders = make_dataloader(args, eval_examples, 'valid')

        # Prepare optimizer and scheduler (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() \
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() \
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                          eps=args.adam_epsilon)
        max_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=max_steps*0.1,
                                                    num_training_steps=max_steps)

        # Start training
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Batch size = {args.train_batch_size}")
        logger.info(f"  Num epochs = {args.num_train_epochs}")

        training_stats = []
        model.zero_grad()
        for epoch in range(args.num_train_epochs):
            training_loop = tqdm(train_dataloader, leave=True)
            epoch_tr_loss = 0

            model.train()
            for batch in training_loop:
                batch = tuple(t.to(args.device) for t in batch)
                # Initialize calculated gradients from previous step
                optimizer.zero_grad()

                # batch: <input_ids, input_masks, labels>
                outputs = model(
                            input_ids=batch[0],
                            attention_mask=batch[1],
                            labels=batch[2],
                          )
                # Batch training loss
                tr_loss = outputs.loss
                epoch_tr_loss += tr_loss.item()

                # Calculate gradients for each parameter
                tr_loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters
                optimizer.step()

                # Update learning rate
                scheduler.step()

                training_loop.set_description(f"Epoch {epoch}")
                training_loop.set_postfix(loss=tr_loss.item())

            # Epoch training loss is the average of the loss of batches in that epoch,
            # i.e., is obtained by dividing by the number of batches.
            epoch_tr_loss = tr_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch}\tTraining Loss: {epoch_tr_loss}")

            # After the completion of one training epoch, measure performance
            # on validation set.
            logger.info('Measuring performance on validation set.')

            # Put the model in evaluation mode--the dropout layers behave
            # differently during evaluation.
            model.eval()

            # Record combined-projects evaluation statistics.
            combined_stats = evaluate(args, model, eval_dataloaders[0], tokenizer)
            logger.info(f"Epoch {epoch}\tValidation Loss: {combined_stats['eval_loss']}")
            logger.info(f"Epoch {epoch}\tCombined Sub-Token Accuracy: {combined_stats['subtoken_accuracy']}")
            logger.info(f"Epoch {epoch}\tCombined Word Accuracy: {combined_stats['word_accuracy']}")

            # Record project-wise evaluation statistics.
            for i, eval_dataloader in enumerate(eval_dataloaders[1:]):
                stats = evaluate(args, model, eval_dataloader, tokenizer)
                logger.info(f"Epoch {epoch}\t{PROJECTS[i].upper()}\t" + \
                            f"Sub-Token Accuracy: {stats['subtoken_accuracy']}\t" +\
                            f"Word Accuracy: {stats['word_accuracy']}")

            epoch_output_dir = Path(output_dir) / f'Epoch_{epoch}'
            epoch_output_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving model to {epoch_output_dir}")
            torch.save(model.state_dict(), str(epoch_output_dir / 'model.ckpt'))

    if args.do_test:
        # Prepare test data loader.
        logger.info('Loading test data.')
        eval_examples = get_examples(args.data_dir, 'pt', 'test')
        logger.info('Constructing data loader for test data.')
        eval_dataloaders = make_dataloader(args, eval_examples, 'test')

        # Start testing
        logger.info("***** Running testing *****")
        logger.info(f"  Num examples = {len(test_examples)}")
        logger.info(f"  Batch size = {args.eval_batch_size}")
        logger.info('Measuring performance on test set.')

        # Put the model in evaluation mode--the dropout layers behave
        # differently during evaluation.
        model.eval()

        # Record combined-projects evaluation statistics.
        combined_stats = evaluate(args, model, eval_dataloaders[0], tokenizer)
        logger.info(f"Epoch {epoch}\tCombined Sub-Token Accuracy: {combined_stats['subtoken_accuracy']}")
        logger.info(f"Epoch {epoch}\tCombined Word Accuracy: {combined_stats['word_accuracy']}")

        # Record project-wise evaluation statistics.
        for i, eval_dataloader in enumerate(eval_dataloaders[1:]):
            stats = evaluate(args, model, eval_dataloader, tokenizer)
            logger.info(f"Epoch {epoch}\t{PROJECTS[i].upper()}\t" + \
                        f"Sub-Token Accuracy: {stats['subtoken_accuracy']}\t" +\
                        f"Word Accuracy: {stats['word_accuracy']}")
