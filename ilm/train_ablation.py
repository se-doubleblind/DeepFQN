from enum import Enum
from collections import defaultdict
import multiprocessing
import os
import re
import json
import pickle
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, CONFIG_NAME, WEIGHTS_NAME

import ilm.constants
import ilm.mask.util
from ilm.mask.hierarchical import MaskHierarchicalType
import ilm.tokenize_util


PROJECTS = ['android', 'gwt', 'hibernate-orm', 'jdk', 'joda-time', 'xstream']


class TargetType(Enum):
  PAD = 0
  CONTEXT = 1
  CONTEXT_SPECIAL = 2
  CONTEXT_INFILL_SEP = 3
  INFILL = 4
  INFILL_SPECIAL = 5
  INFILL_REDUNDANT = 6


class BaseInputExample:
  def __init__(
    self, eid, filename, path_to_source, glob_lines,
  ):
    self.eid = eid
    self.filename = filename
    self.path_to_source = path_to_source
    self.glob_lines = glob_lines


class AblationInputExample:
  def __init__(self, eid, filename, path_to_source, raw_code, full_code, blank_code, answers, resolved_pairs):
    self.eid = eid
    self.filename = filename
    self.path_to_source = path_to_source
    self.raw_code = raw_code
    self.full_code = full_code
    self.blank_code = blank_code
    self.answers = answers
    self.resolved_pairs = resolved_pairs


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def get_train_examples(path_to_data):
  examples = []
  with open(Path(path_to_data) / f"examples_base_train_pt.pkl", 'rb') as fileobj:
    examples += pickle.load(fileobj)

  with open(Path(path_to_data) / f"examples_base_train_ft.pkl", 'rb') as fileobj:
    examples += pickle.load(fileobj)
  return examples


def load_ilm_examples(data_dir, split):
  train_examples = get_train_examples(data_dir)
  val_examples, test_examples = [], []

  for project in PROJECTS:
    data_path = Path(data_dir) / f"{project}.json"
    with open(str(data_path), 'r', encoding='utf-8') as file_obj:
      functions = json.load(file_obj)

    project_examples = []
    for function in functions:
      raw_code = function['code']
      full_code = function['fqn_code']
      blank_code = function['hole_code']
      answers = list(function['pairs'].values())
      resolved_pairs = function['resolved_pairs']

      if blank_code.count('<blank>') != len(answers):
        continue

      project_examples.append(
        AblationInputExample(eid=function['id'], filename=function['file'],
                             path_to_source=function['path_to_source'],
                             raw_code=raw_code, full_code=full_code, blank_code=blank_code,
                             answers=answers, resolved_pairs=resolved_pairs,
        )
      )

    num_train = int(0.8 * len(project_examples))
    num_eval = int(0.1 * len(project_examples))
    val_examples += project_examples[num_train: num_train + num_eval]
    test_examples += project_examples[num_train + num_eval: ]

  if split == 'train':
    return train_examples
  elif split == 'valid':
    return val_examples
  elif split == 'test':
    return test_examples


def doc_and_char_masks_to_input_and_tt(
    doc,
    answers,
    tokenizer,
    start_infill_id,
    end_infill_id,
    mask_id,
    sequence_length):
  try:
    blank_idx = [m.start() for m in re.finditer('<blank>', doc)]
    doc_substrings = doc.split('<blank>')

    doc_tokens, doc_tokens_ids = [], []
    for ss in doc_substrings[:-1]:
      ss_tokens = ilm.tokenize_util.tokenize(ss, tokenizer=tokenizer)
      doc_tokens += ss_tokens + ["<|blank|>"]
      doc_tokens_ids += ilm.tokenize_util.tokens_to_ids(ss_tokens, tokenizer=tokenizer) + [mask_id]

    last_ss_tokens = ilm.tokenize_util.tokenize(doc_substrings[-1], tokenizer=tokenizer)
    doc_tokens += last_ss_tokens
    doc_tokens_ids += ilm.tokenize_util.tokens_to_ids(last_ss_tokens, tokenizer=tokenizer)

    special_ids = set([start_infill_id, end_infill_id, mask_id])
    inputs = np.zeros(sequence_length, dtype=np.uint16)
    tts = np.full(sequence_length, TargetType.PAD.value, dtype=np.uint8)

    # Create example
    example = []

    # (Masked) Context
    # Example: She ate <?> for <?>
    example += doc_tokens_ids

    # Context / answer separator
    context_len = len(example)
    # Example: <S>
    example += [start_infill_id]

    # Answers
    # Example: cereal<E>breakfast<E>
    for answer in answers:
      answer_tokens = ilm.tokenize_util.tokenize(answer, tokenizer=tokenizer)
      answer_tokens_ids = ilm.tokenize_util.tokens_to_ids(answer_tokens, tokenizer=tokenizer)
      example += answer_tokens_ids
      example += [end_infill_id]

    if len(example) > sequence_length:
      example = example[:sequence_length]

    # Find special tokens
    context_special_idxs = [l for l, t in enumerate(example) if l < context_len and t in special_ids]
    infill_special_idxs = [l for l, t in enumerate(example) if l > context_len and t in special_ids]

    # Store example in output array
    if len(example) > 0 and (min(example) < np.iinfo(inputs.dtype).min or max(example) > np.iinfo(inputs.dtype).max):
      raise ValueError('Example cannot be stored in numpy array')
    inputs[:len(example)] = example

    # Store target types in output array
    tts[:context_len] = TargetType.CONTEXT.value
    for l in context_special_idxs:
      tts[l] = TargetType.CONTEXT_SPECIAL.value

    tts[context_len] = TargetType.CONTEXT_INFILL_SEP.value
    tts[context_len+1:len(example)] = TargetType.INFILL.value
    for l in infill_special_idxs:
      tts[l] = TargetType.INFILL_SPECIAL.value


    return inputs, tts
  except Exception as e:
    return None


def masked_dataset_to_inputs_and_tts(
    split,
    tokenizer,
    start_infill_id,
    end_infill_id,
    mask_id,
    args):
  assert split in ['train', 'valid', 'test']
  dataset_examples = load_ilm_examples(args.examples_dir, split)

  if split == 'train':
    sequence_length = args.train_sequence_length

    docs_inputs_and_tts = []
    inputs, tts = [], []
    skipped_ctr = 0
    for example in tqdm(dataset_examples):
      for glob_example in example.glob_lines:
        code = glob_example["globCode"]
        blank = glob_example["blank"]
        doc_inputs_and_tts = doc_and_char_masks_to_input_and_tt(
            code, [blank], tokenizer, start_infill_id, end_infill_id,
            mask_id, sequence_length,
        )
        if doc_inputs_and_tts is not None:
          inputs.append(doc_inputs_and_tts[0])
          tts.append(doc_inputs_and_tts[1])
        else:
          skipped_ctr += 1
    num_examples = len(inputs)
  else:
    sequence_length = args.eval_sequence_length
    num_examples = len(dataset_examples)

    docs_inputs_and_tts = []
    inputs, tts = [], []
    skipped_ctr = 0
    for example in dataset_examples:
      doc_inputs_and_tts = doc_and_char_masks_to_input_and_tt(
          example.blank_code, example.answers, tokenizer, start_infill_id, end_infill_id,
          mask_id, sequence_length,
      )
      if doc_inputs_and_tts is not None:
        inputs.append(doc_inputs_and_tts[0])
        tts.append(doc_inputs_and_tts[1])
      else:
        skipped_ctr += 1

  print(f'Skipped for {skipped_ctr} examples.')
  print(len(inputs), inputs[0].shape)
  inputs = np.asarray(inputs)
  tts = np.asarray(tts)

  return inputs, tts, num_examples


def tts_to_labels(inputs, tts, label_tts):
  selector = torch.zeros_like(inputs, dtype=torch.bool)
  for tt in label_tts:
    selector |= tts == tt.value
  return torch.where(
      selector,
      inputs,
      torch.full_like(inputs, -1))


def train(args):
  # Init device
  n_gpu = torch.cuda.device_count()
  if n_gpu == 0:
    warnings.warn('No GPU detected. Training on CPU will be very slow')
  elif n_gpu > 1:
    warnings.warn('This codebase is not optimized for multi GPU usage')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Lambda for filenames
  example_tag_to_fp = lambda tag: os.path.join(args.examples_dir, '{}.pkl'.format(tag))
  out_fn_to_fp = lambda fn: os.path.join(args.train_dir, fn)

  # Create training dir
  os.makedirs(args.train_dir, exist_ok=True)
  resuming = os.path.exists(out_fn_to_fp('step.pkl'))

  # Create tokenizer
  tokenizer = ilm.tokenize_util.Tokenizer[args.tokenizer_name.upper()]
  if tokenizer == ilm.tokenize_util.Tokenizer.CUSTOM:
    ilm.tokenize_util.set_custom_vocab_fp(args.tokenizer_custom_vocab_fp)

  # Update tokenizer
  base_vocab_size = ilm.tokenize_util.vocab_size(tokenizer)
  start_infill_id = base_vocab_size + 0
  end_infill_id = base_vocab_size + 1
  mask_id = base_vocab_size + 2
  additional_ids_to_tokens = {
      start_infill_id: '<|start-of-infill|>',
      end_infill_id: '<|end-of-infill|>',
      mask_id: '<|blank|>'
  }
  print(additional_ids_to_tokens)
  vocab_size = ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
  with open(out_fn_to_fp('additional_ids_to_tokens.pkl'), 'wb') as f:
    pickle.dump(additional_ids_to_tokens, f)

  # Load training data
  if not args.eval_only:
    print('Loading training data')
    loaded_from_cache = False
    if args.data_cache:
      try:
        train_inputs = np.load(out_fn_to_fp('train_inp.npy'))
        train_tts = np.load(out_fn_to_fp('train_tts.npy'))
        with open(out_fn_to_fp('train_num_docs.pkl'), 'rb') as f:
          train_num_docs = pickle.load(f)
        loaded_from_cache = True
      except:
        pass
    if not loaded_from_cache:
      train_inputs, train_tts, train_num_docs = masked_dataset_to_inputs_and_tts(
          'train',
          tokenizer,
          start_infill_id,
          end_infill_id,
          mask_id,
          args)
      if args.data_cache:
        np.save(out_fn_to_fp('train_inp.npy'), train_inputs)
        np.save(out_fn_to_fp('train_tts.npy'), train_tts)
        with open(out_fn_to_fp('train_num_docs.pkl'), 'wb') as f:
          pickle.dump(train_num_docs, f)
    train_tt_to_count = {TargetType(k):v for k, v in zip(*np.unique(train_tts, return_counts=True))}
    print(train_tt_to_count)
    num_unmasked = train_tt_to_count.get(TargetType.CONTEXT, 0)
    num_masked = train_tt_to_count.get(TargetType.INFILL, 0)
    print('Mask rate (tokens): {:.4f}'.format(num_masked / (num_unmasked + num_masked)))
    print('{} documents, {} examples'.format(train_num_docs, train_inputs.shape[0]))
    print(train_inputs.shape, train_inputs.dtype, train_tts.shape, train_tts.dtype)
    train_data = TensorDataset(
        torch.from_numpy(train_inputs.astype(np.int64)),
        torch.from_numpy(train_tts))
    del train_inputs
    del train_tts

  # Load eval data
  print('Loading validation data')
  loaded_from_cache = False
  if args.data_cache:
    try:
      eval_inputs = np.load(out_fn_to_fp('valid_inp.npy'))
      eval_tts = np.load(out_fn_to_fp('valid_tts.npy'))
      with open(out_fn_to_fp('valid_num_docs.pkl'), 'rb') as f:
        eval_num_docs = pickle.load(f)
      loaded_from_cache = True
    except:
      pass
  if not loaded_from_cache:
    eval_inputs, eval_tts, eval_num_docs = masked_dataset_to_inputs_and_tts(
        'valid',
        tokenizer,
        start_infill_id,
        end_infill_id,
        mask_id,
        args)
    if args.data_cache:
      np.save(out_fn_to_fp('valid_inp.npy'), eval_inputs)
      np.save(out_fn_to_fp('valid_tts.npy'), eval_tts)
      with open(out_fn_to_fp('valid_num_docs.pkl'), 'wb') as f:
        pickle.dump(eval_num_docs, f)
  eval_tt_to_count = {TargetType(k):v for k, v in zip(*np.unique(eval_tts, return_counts=True))}
  print(eval_tt_to_count)
  num_unmasked = eval_tt_to_count.get(TargetType.CONTEXT, 0)
  num_masked = eval_tt_to_count.get(TargetType.INFILL, 0)
  print('Mask rate (tokens): {:.4f}'.format(num_masked / (num_unmasked + num_masked)))
  print('{} documents, {} examples'.format(eval_num_docs, eval_inputs.shape[0]))
  print(eval_inputs.shape, eval_inputs.dtype, eval_tts.shape, eval_tts.dtype)
  eval_data = TensorDataset(
      torch.from_numpy(eval_inputs.astype(np.int64)),
      torch.from_numpy(eval_tts))
  del eval_inputs
  del eval_tts

  # Calculate number of steps to train for (return if we're just pre-cacheing data)
  if args.train_num_epochs is not None:
    train_num_batches = int(float(train_num_docs * args.train_num_epochs) / args.train_batch_size)
    if train_num_batches == 0:
      return
    print('Maximum number of training steps: {}'.format(train_num_batches / args.train_batch_accumulation))

  # Create data iterators
  print('Creating datasets')
  if not args.eval_only:
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

  # Load model
  print('Initializing model...')
  set_random_seed(args.seed)
  if args.model_name in ilm.constants.GPT2_MODEL_NAMES:
    model_type = GPT2LMHeadModel
    cfg_type = GPT2Config
  if resuming:
    print('from saved checkpoint (resuming)')
    model = model_type.from_pretrained(args.train_dir)
  else:
    if args.train_from_scratch:
      print('from scratch')
      cfg = cfg_type.from_pretrained(args.model_name)
      model = model_type(cfg)
    else:
      print('from pretrained checkpoint')
      model = model_type.from_pretrained(args.model_name)
  model.resize_token_embeddings(vocab_size)
  model.to(device)
  model.train()
  print(f'Number of model parameters: {sum(p.numel() for p in model.parameters())}')

  # Reset random seed in case model init triggered RNG

  # Initialize optimizers
  if not args.eval_only:
    params = list(model.named_parameters())
    no_decay = ['bias', 'ln']
    optimizer_grouped_parameters = [
      {
        'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
        'weight_decay': args.train_weight_decay
      },
      {
        'params': [p for n, p in params if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
      }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.train_learning_rate,
        eps=args.train_adam_epsilon)
    if resuming:
      optimizer.load_state_dict(torch.load(out_fn_to_fp('optimizer.pt')))

  # Create global step
  if resuming:
    try:
      with open(out_fn_to_fp('step.pkl'), 'rb') as f:
        step = pickle.load(f)
    except Exception as e:
      if args.eval_only:
        step = None
      else:
        raise e
  else:
    step = 0

  if args.eval_only:
    print('Evaluating')
    model.eval()

    eval_start = time.time()
    eval_token_counts = defaultdict(int)
    eval_token_loss_sums = defaultdict(float)
    for i, eval_batch in enumerate(eval_dataloader):
      with torch.no_grad():
        eval_inputs, eval_tts = tuple(t.to(device) for t in eval_batch)
        eval_logits, _ = model(eval_inputs)
        eval_logits_relevant = eval_logits[:, :-1].contiguous().view(-1, eval_logits.shape[-1])

        for tag, tts in [
            ('context', [TargetType.CONTEXT]),
            ('infill', [TargetType.INFILL, TargetType.INFILL_SPECIAL]),
            ('infill_textonly', [TargetType.INFILL])]:
          eval_labels = tts_to_labels(eval_inputs, eval_tts, tts)
          eval_labels_relevant = eval_labels[:, 1:]
          eval_labels_relevant_count = (eval_labels_relevant != -1).long().sum().item()
          eval_labels_loss = F.cross_entropy(
              eval_logits_relevant,
              eval_labels_relevant.contiguous().view(-1),
              ignore_index=-1).item()
          eval_token_counts[tag] += eval_labels_relevant_count
          eval_token_loss_sums[tag] += eval_labels_loss * eval_labels_relevant_count

    eval_dict = {}
    for tag, count in eval_token_counts.items():
      loss = eval_token_loss_sums[tag]
      if count > 0:
        loss /= count
      eval_dict['eval_{}_count'.format(tag)] = count
      eval_dict['eval_{}_loss'.format(tag)] = loss
      eval_dict['eval_{}_ppl'.format(tag)] = np.exp(loss)
    eval_dict['eval_time'] = time.time() - eval_start

    print('-' * 80)
    if step is not None:
      print('(Step {}) Eval'.format(step))
    for k, v in eval_dict.items():
      print('{}: {}'.format(k, v))

  else:
    print('Training')
    set_random_seed(args.seed)
    best_eval_loss = None
    num_save = -1
    num_summary = -1
    num_batches_complete = step * args.train_batch_accumulation
    start = time.time()
    while True:
      print(num_batches_complete, train_num_batches)
      if args.train_num_epochs is not None and num_batches_complete >= train_num_batches:
        break

      for batch in tqdm(train_dataloader):
        if args.train_num_epochs is not None and num_batches_complete >= train_num_batches:
          break

        elapsed = time.time() - start

        # Evaluate
        if int(elapsed / args.train_eval_secs) > num_save:
          num_save = int(elapsed / args.train_eval_secs)

          model.eval()

          eval_start = time.time()
          eval_token_counts = defaultdict(int)
          eval_token_loss_sums = defaultdict(float)

          eval_predictions = []
          for i, eval_batch in enumerate(eval_dataloader):
            with torch.no_grad():
              eval_inputs, eval_tts = tuple(t.to(device) for t in eval_batch)
              eval_outputs = model(eval_inputs)
              eval_logits = eval_outputs.logits
              eval_logits_relevant = eval_logits[:, :-1].contiguous().view(-1, eval_logits.shape[-1])

              for tag, tts in [
                  ('context', [TargetType.CONTEXT]),
                  ('context_true', [TargetType.INFILL, TargetType.INFILL_SPECIAL]),
                  ('infill', [TargetType.INFILL, TargetType.INFILL_SPECIAL]),
                  ('infill_textonly', [TargetType.INFILL])]:

                eval_labels = tts_to_labels(eval_inputs, eval_tts, tts)
                eval_labels_relevant = eval_labels[:, 1:]

                eval_labels_relevant_count = (eval_labels_relevant != -1).long().sum().item()
                eval_labels_loss = F.cross_entropy(
                    eval_logits_relevant,
                    eval_labels_relevant.contiguous().view(-1),
                    ignore_index=-1).item()
                eval_token_counts[tag] += eval_labels_relevant_count
                eval_token_loss_sums[tag] += eval_labels_loss * eval_labels_relevant_count

              # # Compute accuracy
              # eval_pred_logits = torch.argmax(eval_logits, dim=2)
              # eval_pred_labels = tts_to_labels(eval_pred_logits, eval_tts,
              #                                  [TargetType.INFILL, TargetType.INFILL_SPECIAL])
              # eval_labels = tts_to_labels(eval_inputs, eval_tts,
              #                             [TargetType.CONTEXT_INFILL_SEP, TargetType.INFILL, TargetType.INFILL_SPECIAL])

              # for j, sample_pred_labels in enumerate(eval_pred_labels):
              #   pj = list(sample_pred_labels.cpu().numpy())
              #   preds = ilm.tokenize_util.decode([0 if t == -1 else t for t in pj], tokenizer)
              #   preds = preds.replace('!', '')
              #   print(f'Prediction: {preds}')

              #   tj = list(eval_labels[j].cpu().numpy())
              #   true = ilm.tokenize_util.decode([0 if t == -1 else t for t in tj], tokenizer)
              #   true = true.replace('!', '')
              #   print(f'True: {true}')
              #   print('**********************')
              #   print()
              #   eval_predictions.append({
              #     'Prediction': preds,
              #     'True': true,
              #   })

          # print(f'Actual number of evaluation samples: {len(eval_dataloader) * 8}')
          # print(f'Number of evaluation samples: {len(eval_predictions)}')
          # with open(f'{args.train_dir}/predictions_{int(time.time())}.json', 'w') as f:
          #   json.dump(eval_predictions, f)

          eval_dict = {}
          for tag, count in eval_token_counts.items():
            loss = eval_token_loss_sums[tag]
            if count > 0:
              loss /= count
            eval_dict['eval_{}_count'.format(tag)] = count
            eval_dict['eval_{}_loss'.format(tag)] = loss
          eval_dict['eval_time'] = time.time() - eval_start

          print('-' * 80)
          print('(Step {}) Eval'.format(step))
          for k, v in eval_dict.items():
            print('{}: {}'.format(k, v))

          if best_eval_loss is None or eval_dict['eval_infill_loss'] < best_eval_loss:
            print('Saving')
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.config.to_json_file(out_fn_to_fp(CONFIG_NAME))
            torch.save(model_to_save.state_dict(), out_fn_to_fp(WEIGHTS_NAME))
            torch.save(optimizer.state_dict(), out_fn_to_fp('optimizer.pt'))
            with open(out_fn_to_fp('step.pkl'), 'wb') as f:
              pickle.dump(step, f)
            best_eval_loss = eval_dict['eval_infill_loss']

          model.train()

        # Train
        inputs, tts = tuple(t.to(device) for t in batch)
        # TODO: Option to train on CONTEXT_SPECIAL?
        labels_context = tts_to_labels(inputs, tts, [TargetType.CONTEXT])
        labels_infill = tts_to_labels(inputs, tts, [TargetType.INFILL, TargetType.INFILL_SPECIAL])

        outputs = model(inputs)
        logits = outputs.logits
        logits_relevant = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
        loss_context = F.cross_entropy(
            logits_relevant,
            labels_context[:, 1:].contiguous().view(-1),
            ignore_index=-1)
        loss_infill = F.cross_entropy(
            logits_relevant,
            labels_infill[:, 1:].contiguous().view(-1),
            ignore_index=-1)

        loss_context_item = loss_context.item()
        loss_infill_item = loss_infill.item()

        loss = loss_infill
        if args.train_context:
          loss += loss_context

        if args.train_batch_accumulation != 1:
          loss /= float(args.train_batch_accumulation)
        loss.backward()

        # Summarize
        if int(elapsed / args.train_summary_secs) > num_summary:
          num_summary = int(elapsed / args.train_summary_secs)

          print('-' * 80)
          print('(Step {}) Summary'.format(step))
          print(loss_context_item)
          print(loss_infill_item)
          with torch.no_grad():
            for t in inputs, labels_context, labels_infill:
              t0 = list(t[0].cpu().numpy())
#              print('-' * 40)
#              print(t0)
            for t in inputs, labels_context, labels_infill:
              t0 = list(t[0].cpu().numpy())
#              print('-' * 40)
#              print(ilm.tokenize_util.decode([0 if t == -1 else t for t in t0], tokenizer))

        if ((num_batches_complete + 1) % args.train_batch_accumulation) == 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.train_max_grad_norm)
          optimizer.step()
          optimizer.zero_grad()
          step += 1

        num_batches_complete += 1


if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()

  parser.add_argument('train_dir', type=str)
  parser.add_argument('examples_dir', type=str)
  parser.add_argument('--seed', type=int)

  tokenizer_args = parser.add_argument_group('Tokenizer')
  tokenizer_args.add_argument('--tokenizer_name', type=str, choices=[t.name.lower() for t in ilm.tokenize_util.Tokenizer])
  tokenizer_args.add_argument('--tokenizer_custom_vocab_fp', type=str)

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_no_cache', action='store_false', dest='data_cache')

  model_args = parser.add_argument_group('Model')
  model_args.add_argument('--model_name', type=str, choices=ilm.constants.GPT2_MODEL_NAMES)

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_examples_tag', type=str)
  train_args.add_argument('--train_num_epochs', type=int)
  train_args.add_argument('--train_from_scratch', action='store_true', dest='train_from_scratch')
  train_args.add_argument('--train_batch_size', type=int)
  train_args.add_argument('--train_batch_accumulation', type=int)
  train_args.add_argument('--train_sequence_length', type=int)
  train_args.add_argument('--train_eval_secs', type=float)
  train_args.add_argument('--train_summary_secs', type=float)
  train_args.add_argument('--train_minimal_supervision', action='store_false', dest='train_context')
  train_args.add_argument('--train_learning_rate', type=float)
  train_args.add_argument('--train_weight_decay', type=float)
  train_args.add_argument('--train_adam_epsilon', type=float)
  train_args.add_argument('--train_max_grad_norm', type=float)

  eval_args = parser.add_argument_group('Eval')
  eval_args.add_argument('--eval_only', action='store_true', dest='eval_only')
  eval_args.add_argument('--eval_examples_tag', type=str)
  eval_args.add_argument('--eval_batch_size', type=int)
  eval_args.add_argument('--eval_sequence_length', type=int)

  parser.set_defaults(
      seed=42,
      tokenizer_name='gpt2',
      tokenizer_custom_vocab_fp=None,
      data_cache=True,
      model_name='gpt2',
      train_examples_tag='train',
      train_num_epochs=None,
      train_from_scratch=False,
      train_batch_size=16,
      train_batch_accumulation=3,
      train_sequence_length=512,
      train_eval_secs=360,
      train_summary_secs=360,
      train_context=True,
      train_learning_rate=1e-5,
      train_weight_decay=0.,
      train_adam_epsilon=1e-8,
      train_max_grad_norm=1.,
      eval_only=False,
      eval_examples_tag='valid',
      eval_batch_size=16,
      eval_sequence_length=512)
  
  args = parser.parse_args()

  if args.seed is None:
    args.seed = random.randint(0, 1e6)
  print('Random seed {}'.format(args.seed))

  train(args)
