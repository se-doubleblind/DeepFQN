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
from transformers.generation import StoppingCriteriaList, StoppingCriteria

import ilm.constants
import ilm.mask.util
from ilm.mask.hierarchical import MaskHierarchicalType
import ilm.tokenize_util
from compute_metrics import compute_accuracy, compute_accuracy2


PROJECTS = ['android', 'gwt', 'hibernate-orm', 'jdk', 'joda-time', 'xstream']

EVAL = 'valid'


class TargetType(Enum):
  PAD = 0
  CONTEXT = 1
  CONTEXT_SPECIAL = 2
  CONTEXT_INFILL_SEP = 3
  INFILL = 4
  INFILL_SPECIAL = 5
  INFILL_REDUNDANT = 6


class InputExample:
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


def generate(
  model,
  tokenizer,
  prompt_ids,
  mask_id,
  end_infill_id,
  top_p=0.8,
  temperature=1.,
):
  model.eval()
  filter_value = -float("Inf")

  with torch.no_grad():
    num_blanks =  torch.count_nonzero(prompt_ids == mask_id)
    stop_generation = False
    stop_ctr = 0
    generated = prompt_ids.clone().detach().unsqueeze(0)

    while not stop_generation:
      outputs = model(generated, labels=generated)
      loss, logits = outputs.loss, outputs.logits
      logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

      sorted_logits, sorted_indices = torch.sort(logits, descending=True)
      cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

      sorted_indices_to_remove = cumulative_probs > top_p
      sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
          ..., :-1
      ].clone()
      sorted_indices_to_remove[..., 0] = 0

      indices_to_remove = sorted_indices[sorted_indices_to_remove]
      logits[:, indices_to_remove] = filter_value

      next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
      generated = torch.cat((generated, next_token), dim=1)

      if next_token == end_infill_id:
        stop_ctr += 1
        if stop_ctr == num_blanks:
          stop_generation = True

  answers_ids = generated.tolist()[0][prompt_ids.shape[0]:]
  answers = ilm.tokenize_util.decode(answers_ids, tokenizer)                
  return answers_ids, answers


def load_ilm_examples(data_dir, split):
  try:
    with open(str(Path(data_dir) / f'examples_{split}.pkl'), 'rb') as handler:
      examples = pickle.load(handler)
    return examples

  except FileNotFoundError:
    train_examples, val_examples, test_examples = [], [], []
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
          InputExample(eid=function['id'], filename=function['file'],
                       path_to_source=function['path_to_source'],
                       raw_code=raw_code, full_code=full_code, blank_code=blank_code,
                       answers=answers, resolved_pairs=resolved_pairs,
          )
        )

      num_train = int(0.8 * len(project_examples))
      num_eval = int(0.1 * len(project_examples))
      train_examples += project_examples[: num_train]
      val_examples += project_examples[num_train: num_train + num_eval]
      test_examples += project_examples[num_train + 2 * num_eval: ]

    with open(str(Path(data_dir) / f"examples_train.pkl"), 'wb') as handler:
      pickle.dump(train_examples, handler)

    with open(str(Path(data_dir) / f"examples_valid.pkl"), 'wb') as handler:
      pickle.dump(val_examples, handler)

    with open(str(Path(data_dir) / f"examples_test.pkl"), 'wb') as handler:
      pickle.dump(test_examples, handler)

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
  if split == 'train':
    sequence_length = args.train_sequence_length
    max_num_examples = args.train_max_num_examples
  else:
    sequence_length = args.eval_sequence_length
    max_num_examples = args.eval_max_num_examples

  dataset_examples = load_ilm_examples(args.examples_dir, split)
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

  # TODO: Don't bother doing all the work if we're not going to use it
  if max_num_examples is not None:
    set_random_seed(args.seed)
    example_ids = random.sample(list(range(inputs.shape[0])), max_num_examples)
    inputs = np.take(inputs, example_ids, axis=0)
    tts = np.take(tts, example_ids, axis=0)

  return inputs, tts, num_examples


def tts_to_labels(inputs, tts, label_tts):
  selector = torch.zeros_like(inputs, dtype=torch.bool)
  for tt in label_tts:
    selector |= tts == tt.value
  return torch.where(
      selector,
      inputs,
      torch.full_like(inputs, -1))


def evaluate(args):
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

  # Initialize resuming flag
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

  # Load eval data
  print(f'Loading {EVAL} data')
  loaded_from_cache = False
  if args.data_cache:
    try:
      eval_inputs = np.load(out_fn_to_fp(f'{EVAL}_inp.npy'))
      eval_tts = np.load(out_fn_to_fp(f'{EVAL}_tts.npy'))
      with open(out_fn_to_fp(f'{EVAL}_num_docs.pkl'), 'rb') as f:
        eval_num_docs = pickle.load(f)
      loaded_from_cache = True
    except:
      pass

  if not loaded_from_cache:
    eval_inputs, eval_tts, eval_num_docs = masked_dataset_to_inputs_and_tts(
        EVAL,
        tokenizer,
        start_infill_id,
        end_infill_id,
        mask_id,
        args)

    if args.data_cache:
      np.save(out_fn_to_fp(f'{EVAL}_inp.npy'), eval_inputs)
      np.save(out_fn_to_fp(f'{EVAL}_tts.npy'), eval_tts)
      with open(out_fn_to_fp(f'{EVAL}_num_docs.pkl'), 'wb') as f:
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

  # Create data iterators
  print('Creating datasets')
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
    return

  model.resize_token_embeddings(vocab_size)
  model.to(device)
  print(f'Number of model parameters: {sum(p.numel() for p in model.parameters())}')

  # Create global step
  if resuming:
    try:
      with open(out_fn_to_fp('step.pkl'), 'rb') as f:
        step = pickle.load(f)
    except Exception as e:
        step = None
  else:
    step = 0

  print('Evaluating')
  model.eval()

  eval_start = time.time()
  eval_token_counts = defaultdict(int)
  eval_token_loss_sums = defaultdict(float)

  eval_predictions = []
  for i, eval_batch in tqdm(enumerate(eval_dataloader)):
    with torch.no_grad():
      eval_inputs, eval_tts = tuple(t.to(device) for t in eval_batch)

      for j, sample_inputs in enumerate(eval_inputs):
        _sample_inputs = tts_to_labels(sample_inputs, eval_tts[j], [TargetType.CONTEXT, TargetType.CONTEXT_SPECIAL, TargetType.CONTEXT_INFILL_SEP])
        _sample_inputs = _sample_inputs[_sample_inputs != -1]
        sample_labels = tts_to_labels(sample_inputs, eval_tts[j], [TargetType.INFILL, TargetType.INFILL_SPECIAL])

        answers_ids, answers = generate(model, tokenizer, _sample_inputs, mask_id, end_infill_id)

        tj = list(sample_labels.cpu().numpy())
        true_st = [t for t in tj if t != -1]
        # true_st = [t for t in sample_labels.tolist() if t != -1]
        true = ilm.tokenize_util.decode(true_st, tokenizer)

        eval_predictions.append({
          'Prediction_Subtokens': answers_ids,
          'Prediction': answers,
          'True_Subtokens': true_st,
          'True': true,
        })

  print(f'Actual number of evaluation samples: {len(eval_dataloader) * 8}')
  print(f'Number of evaluation samples: {len(eval_predictions)}')

  eval_dict = {}
  eval_dict['eval_time'] = time.time() - eval_start

  print('-' * 80)
  if step is not None:
    print('(Step {}) Eval'.format(step))
  for k, v in eval_dict.items():
    print('{}: {}'.format(k, v))

  compute_accuracy2(eval_predictions)


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
  train_args.add_argument('--train_max_num_examples', type=int)
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
  eval_args.add_argument('--eval_examples_tag', type=str)
  eval_args.add_argument('--eval_max_num_examples', type=int)
  eval_args.add_argument('--eval_batch_size', type=int)
  eval_args.add_argument('--eval_sequence_length', type=int)

  parser.set_defaults(
      seed=42,
      tokenizer_name='gpt2',
      tokenizer_custom_vocab_fp=None,
      data_cache=True,
      model_name='gpt2',
      train_examples_tag='train',
      train_max_num_examples=None,
      train_num_epochs=None,
      train_from_scratch=False,
      train_batch_size=8,
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
      eval_examples_tag=EVAL,
      eval_max_num_examples=None,
      eval_batch_size=8,
      eval_sequence_length=512)
  
  args = parser.parse_args()

  if args.seed is None:
    args.seed = random.randint(0, 1e6)
  print('Random seed {}'.format(args.seed))

  evaluate(args)
