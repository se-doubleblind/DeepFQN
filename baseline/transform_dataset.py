import re
import json
import random
import pickle
import argparse
from pathlib import Path
from copy import deepcopy

from tqdm import tqdm

import numpy as np

import torch


CONTEXT_LENGTH = 2
PRETRAIN_PCT = 0.3
VALID_AST_NODES = ['CLASS_INSTANCE_CREATION', 'FIELD_ACCESS','METHOD_INVOCATION',
                   'SINGLE_VARIABLE_DECLARATION']
PROJECTS = ['android', 'gwt', 'hibernate-orm', 'jdk', 'joda-time', 'xstream']


class InputExample:
  def __init__(
    self, eid, filename, path_to_source, raw_code, full_code,
    blank_code, answers, resolved_pairs,
  ):
    self.eid = eid
    self.filename = filename
    self.path_to_source = path_to_source
    self.raw_code = raw_code
    self.full_code = full_code
    self.blank_code = blank_code
    self.answers = answers
    self.resolved_pairs = resolved_pairs

class BaseInputExample:
  def __init__(
    self, eid, filename, path_to_source, glob_lines,
  ):
    self.eid = eid
    self.filename = filename
    self.path_to_source = path_to_source
    self.glob_lines = glob_lines


def _test_transform_example():
  examples = [
    InputExample(
      eid=None, filename=None, path_to_source=None, full_code=None,
      raw_code=("public void testToDateTime_Chronology() {"
                "\nInstant test = new Instant(TEST_TIME1);"
                "\nDateTime result = test.toDateTime(ISOChronology.getInstance());"
                "\nassertEquals(test.getMillis(), result.getMillis());"
                "\nassertEquals(ISOChronology.getInstance(), "
                "result.getChronology());\ntest = new Instant(TEST_TIME1);"
                "\nresult = test.toDateTime(GregorianChronology.getInstance(PARIS));"
                "\nassertEquals(test.getMillis(), result.getMillis());"
                "\nassertEquals(GregorianChronology.getInstance(PARIS), "
                "result.getChronology());\ntest = new Instant(TEST_TIME1);"
                "\nresult = test.toDateTime((Chronology) null);"
                "\nassertEquals(ISOChronology.getInstance(), result.getChronology());\n}"),
      blank_code=("public void testToDateTime_Chronology() {"
                  "\nInstant test = new Instant(TEST_TIME1);"
                  "\nDateTime result = test.toDateTime(ISOChronology.getInstance());"
                  "\nassertEquals(test.getMillis(), result.getMillis());"
                  "\nassertEquals(ISOChronology.getInstance(), "
                  "result.getChronology());\ntest = new <blank>.Instant(TEST_TIME1);"
                  "\nresult = <blank>.toDateTime(<blank>.getInstance(PARIS));"
                  "\nassertEquals(test.getMillis(), result.getMillis());"
                  "\nassertEquals(GregorianChronology.getInstance(PARIS), "
                  "result.getChronology());\ntest = new <blank>.Instant(TEST_TIME1);"
                  "\nresult = <blank>.toDateTime((Chronology)null);"
                  "\nassertEquals(ISOChronology.getInstance(), result.getChronology());\n}"),
      answers=["org.joda.time",
               "org.joda.time.base.AbstractInstant",
               "org.joda.time.chrono.GregorianChronology",
               "org.joda.time",
               "org.joda.time.base.AbstractInstant"],
      resolved_pairs={
        "0": ["org.joda.time.Instant", "CLASS_INSTANCE_CREATION"],
        "1": ["org.joda.time.base.AbstractInstant", "METHOD_INVOCATION"],
        "2": ["org.joda.time.chrono.GregorianChronology", "METHOD_INVOCATION"],
        "3": ["org.joda.time.Instant", "CLASS_INSTANCE_CREATION"],
        "4": ["org.joda.time.base.AbstractInstant", "METHOD_INVOCATION"],
      },
    ),
    InputExample(
      eid=None, filename=None, path_to_source=None, full_code=None,
      raw_code=("@Override\npublic boolean equals(Object obj) "
                "{\nif (this == obj) {\nreturn true;\n}\nif (obj "
                "instanceof StandardDurationFieldType) "
                "{\nreturn iOrdinal == ((StandardDurationFieldType) "
                "obj).iOrdinal;\n}\nreturn false;\n}"),
      blank_code=("@Override\npublic boolean equals(Object obj) "
                  "{\nif (this == obj) {\nreturn true;\n}\nif (obj "
                  "instanceof <blank>.StandardDurationFieldType) "
                  "{\nreturn iOrdinal == <blank>.iOrdinal;\n}\nreturn false;\n}"),
      answers=["org.joda.time.DurationFieldType",
               "org.joda.time.DurationFieldType.StandardDurationFieldType"],
      resolved_pairs={
        "0": ["org.joda.time.DurationFieldType.StandardDurationFieldType",
              "INSTANCEOF_EXPRESSION"],
        "1": ["org.joda.time.DurationFieldType.StandardDurationFieldType.iOrdinal",
              "FIELD_ACCESS"],
      },
    ),
  ]

  lines_train = {
    0: [
      [{"globCode": "\ntest = new <blank>.Instant(TEST_TIME1)",
        "blank": "org.joda.time",
        "resolved": ["org.joda.time.Instant", "CLASS_INSTANCE_CREATION"]},
       {"globCode": "\nresult = <blank>.toDateTime(.getInstance(PARIS))",
        "blank": "org.joda.time.base.AbstractInstant",
        "resolved": ["org.joda.time.base.AbstractInstant", "METHOD_INVOCATION"]},
       {"globCode": "\nresult = .toDateTime(<blank>.getInstance(PARIS))",
        "blank": "org.joda.time.chrono.GregorianChronology",
        "resolved": ["org.joda.time.chrono.GregorianChronology", "METHOD_INVOCATION"]},
       {"globCode": "\ntest = new <blank>.Instant(TEST_TIME1)",
        "blank": "org.joda.time",
        "resolved": ["org.joda.time.Instant", "CLASS_INSTANCE_CREATION"]},
       {"globCode": "\nresult = <blank>.toDateTime((Chronology)null)",
        "blank": "org.joda.time.base.AbstractInstant",
        "resolved": ["org.joda.time.base.AbstractInstant", "METHOD_INVOCATION"]}],
      [{"globCode": ("\n}\nif (obj instanceof .StandardDurationFieldType) "
                     "{\nreturn iOrdinal == <blank>.iOrdinal"),
        "blank": "org.joda.time.DurationFieldType.StandardDurationFieldType",
        "resolved": ["org.joda.time.DurationFieldType.StandardDurationFieldType.iOrdinal", "FIELD_ACCESS"]}],
    ],
    1: [
      [{"globCode": ("\nassertEquals(ISOChronology.getInstance(),"
                     " result.getChronology());\ntest = new <blank>.Instant(TEST_TIME1);"
                     "\nresult = test.toDateTime(GregorianChronology.getInstance(PARIS))"),
        "blank": "org.joda.time",
        "resolved": ["org.joda.time.Instant", "CLASS_INSTANCE_CREATION"]},
       {"globCode": ("\ntest = new Instant(TEST_TIME1);"
                     "\nresult = <blank>.toDateTime(.getInstance(PARIS));"
                     "\nassertEquals(test.getMillis(), result.getMillis())"),
        "blank": "org.joda.time.base.AbstractInstant",
        "resolved": ["org.joda.time.base.AbstractInstant", "METHOD_INVOCATION"]},
       {"globCode": ("\ntest = new Instant(TEST_TIME1);"
                     "\nresult = .toDateTime(<blank>.getInstance(PARIS));"
                     "\nassertEquals(test.getMillis(), result.getMillis())"),
        "blank": "org.joda.time.chrono.GregorianChronology",
        "resolved": ["org.joda.time.chrono.GregorianChronology", "METHOD_INVOCATION"]},
       {"globCode": ("\nassertEquals(GregorianChronology.getInstance(PARIS),"
                     " result.getChronology());\ntest = new <blank>.Instant(TEST_TIME1);"
                     "\nresult = test.toDateTime((Chronology) null)"),
        "blank": "org.joda.time",
        "resolved": ["org.joda.time.Instant", "CLASS_INSTANCE_CREATION"]},
       {"globCode": ("\ntest = new Instant(TEST_TIME1);\nresult = "
                     "<blank>.toDateTime((Chronology)null);"
                     "\nassertEquals(ISOChronology.getInstance(), result.getChronology())"),
        "blank": "org.joda.time.base.AbstractInstant",
        "resolved": ["org.joda.time.base.AbstractInstant", "METHOD_INVOCATION"]}],
      [{"globCode": ("@Override\npublic boolean equals(Object obj) {\nif (this == obj) {\nreturn true;"
                    "\n}\nif (obj instanceof .StandardDurationFieldType) "
                    "{\nreturn iOrdinal == <blank>.iOrdinal;\n}\nreturn false"),
        "blank": "org.joda.time.DurationFieldType.StandardDurationFieldType",
        "resolved": ["org.joda.time.DurationFieldType.StandardDurationFieldType.iOrdinal", "FIELD_ACCESS"]}],
    ],
  }

  lines_eval = {
    0:   [
      [{"globCode": "\ntest = new <blank>.Instant(TEST_TIME1)",
        "blank": "org.joda.time",
        "resolved": ["org.joda.time.Instant", "CLASS_INSTANCE_CREATION"]},
       {"globCode": "\nresult = <blank>.toDateTime(.getInstance(PARIS))",
        "blank": "org.joda.time.base.AbstractInstant",
        "resolved": ["org.joda.time.base.AbstractInstant", "METHOD_INVOCATION"]},
       {"globCode": "\nresult = .toDateTime(<blank>.getInstance(PARIS))",
        "blank": "org.joda.time.chrono.GregorianChronology",
        "resolved": ["org.joda.time.chrono.GregorianChronology", "METHOD_INVOCATION"]},
       {"globCode": "\ntest = new <blank>.Instant(TEST_TIME1)",
        "blank": "org.joda.time",
        "resolved": ["org.joda.time.Instant", "CLASS_INSTANCE_CREATION"]},
       {"globCode": "\nresult = <blank>.toDateTime((Chronology)null)",
        "blank": "org.joda.time.base.AbstractInstant",
        "resolved": ["org.joda.time.base.AbstractInstant", "METHOD_INVOCATION"]}],
      [{"globCode": ("\n}\nif (obj instanceof <blank>.StandardDurationFieldType) "
                     "{\nreturn iOrdinal == this.iOrdinal"),
        "blank": "org.joda.time.DurationFieldType",
        "resolved": ["org.joda.time.DurationFieldType.StandardDurationFieldType", "INSTANCEOF_EXPRESSION"]},
       {"globCode": ("\n}\nif (obj instanceof .StandardDurationFieldType) "
                     "{\nreturn iOrdinal == <blank>.iOrdinal"),
        "blank": "org.joda.time.DurationFieldType.StandardDurationFieldType",
        "resolved": ["org.joda.time.DurationFieldType.StandardDurationFieldType.iOrdinal", "FIELD_ACCESS"]}],
    ],
    1: [
      [{"globCode": ("\nassertEquals(ISOChronology.getInstance(),"
                     " result.getChronology());\ntest = new <blank>.Instant(TEST_TIME1);"
                     "\nresult = test.toDateTime(GregorianChronology.getInstance(PARIS))"),
        "blank": "org.joda.time",
        "resolved": ["org.joda.time.Instant", "CLASS_INSTANCE_CREATION"]},
       {"globCode": ("\ntest = new Instant(TEST_TIME1);"
                     "\nresult = <blank>.toDateTime(.getInstance(PARIS));"
                     "\nassertEquals(test.getMillis(), result.getMillis())"),
        "blank": "org.joda.time.base.AbstractInstant",
        "resolved": ["org.joda.time.base.AbstractInstant", "METHOD_INVOCATION"]},
       {"globCode": ("\ntest = new Instant(TEST_TIME1);"
                     "\nresult = .toDateTime(<blank>.getInstance(PARIS));"
                     "\nassertEquals(test.getMillis(), result.getMillis())"),
        "blank": "org.joda.time.chrono.GregorianChronology",
        "resolved": ["org.joda.time.chrono.GregorianChronology", "METHOD_INVOCATION"]},
       {"globCode": ("\nassertEquals(GregorianChronology.getInstance(PARIS),"
                     " result.getChronology());\ntest = new <blank>.Instant(TEST_TIME1);"
                     "\nresult = test.toDateTime((Chronology) null)"),
        "blank": "org.joda.time",
        "resolved": ["org.joda.time.Instant", "CLASS_INSTANCE_CREATION"]},
       {"globCode": ("\ntest = new Instant(TEST_TIME1);\nresult = "
                     "<blank>.toDateTime((Chronology)null);"
                     "\nassertEquals(ISOChronology.getInstance(), result.getChronology())"),
        "blank": "org.joda.time.base.AbstractInstant",
        "resolved": ["org.joda.time.base.AbstractInstant", "METHOD_INVOCATION"]}],
      [{"globCode": ("@Override\npublic boolean equals(Object obj) {\nif (this == obj) {\nreturn true;"
                     "\n}\nif (obj instanceof <blank>.StandardDurationFieldType) "
                     "{\nreturn iOrdinal == this.iOrdinal;\n}\nreturn false"),
        "blank": "org.joda.time.DurationFieldType",
        "resolved": ["org.joda.time.DurationFieldType.StandardDurationFieldType", "INSTANCEOF_EXPRESSION"]},
       {"globCode": ("@Override\npublic boolean equals(Object obj) {\nif (this == obj) {\nreturn true;"
                     "\n}\nif (obj instanceof .StandardDurationFieldType) "
                     "{\nreturn iOrdinal == <blank>.iOrdinal;\n}\nreturn false"),
        "blank": "org.joda.time.DurationFieldType.StandardDurationFieldType",
        "resolved": ["org.joda.time.DurationFieldType.StandardDurationFieldType.iOrdinal", "FIELD_ACCESS"]}],
    ],
  }

  for _id, example in enumerate(examples):
    for split, lines in zip(['train', 'eval'], [lines_train, lines_eval]):
      for context_length in [0, 1]:
        glob_example = transform_example(example, split, context_length)
        glob_lines = lines[context_length]

        # Test for number of data points created for an input example.
        assert len(glob_example) == len(glob_lines[_id])

        for glob_id, glob in enumerate(glob_example):
          # Test (1) glob-code.
          assert glob['globCode'] == glob_lines[_id][glob_id]['globCode']

          # Test (2) answer.
          assert glob['blank'] == glob_lines[_id][glob_id]['blank']

          # Test (3) resolved FQN.
          assert glob['resolved'][0] == glob_lines[_id][glob_id]['resolved'][0]

          # Test (4) AST node type.
          assert glob['resolved'][1] == glob_lines[_id][glob_id]['resolved'][1]


def trim_fqn(resolved, fqn, node_type):
  if node_type in [
    "ARRAY_CREATION", "CAST_EXPRESSION", "CLASS_INSTANCE_CREATION",
    "INSTANCEOF_EXPRESSION", "METHOD_INVOCATION"]:
    unresolved = ""
  elif node_type in ["CONSTRUCTOR_INVOCATION", "FIELD_ACCESS"]:
    unresolved = "this"
  elif node_type in ["SINGLE_VARIABLE_DECLARATION"]:
    unresolved = resolved.replace(fqn, "")[1:]
  elif node_type in [
    "SUPER_CONSTRUCTOR_INVOCATION", "SUPER_FIELD_ACCESS", "SUPER_METHOD_INVOCATION"]:
    unresolved = "super"
  elif node_type == "THROW_STATEMENT":
    unresolved = resolved.split(".")[-1]
  return unresolved


def get_test_examples(data_dir):
  examples = []
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
    examples += project_examples[num_train + num_eval: ]
  return examples


def process_line(line, line_num, code_lines, blank_code_lines, answers,
                 resolved_pairs, blanks_idx, blank_ctr, context_length):
  if line_num - context_length < 0:
    pre = code_lines[:line_num]
  else:
    pre = code_lines[line_num-context_length: line_num]
  cur = blank_code_lines[line_num]
  post = code_lines[line_num+1: line_num+context_length+1]

  unresolved = []
  for _id, blank_idx in enumerate(blanks_idx):
    resolved, node_type = resolved_pairs[blank_ctr + _id]
    fqn = answers[blank_ctr + _id]
    unresolved.append(trim_fqn(resolved, fqn, node_type))

  glob_lines = []
  for _id, blank_idx in enumerate(blanks_idx):
    joins = deepcopy(unresolved)
    joins[_id] = '<blank>'

    new_cur = ""
    cur_splits = cur.split("<blank>")
    for cur_split_id, cur_split in enumerate(cur_splits[:-1]):
      new_cur += cur_split + joins[cur_split_id]
    new_cur += cur_splits[-1]

    glob_lines.append(
      {
        'globCode': ';'.join(pre + [new_cur] + post),
        'blank': answers[blank_ctr + _id],
        'resolved': resolved_pairs[blank_ctr + _id],
      }
    )
  return glob_lines


def transform_example(example, split, context_length):
  if split == 'train':
    answers, resolved_pairs, joins = [], [], []
    for _id, (resolved, node_type) in enumerate(example.resolved_pairs.values()):
      if node_type in VALID_AST_NODES:
        answers.append(example.answers[_id])
        resolved_pairs.append(list(example.resolved_pairs.values())[_id])
        joins.append('<blank>')
      else:
        resolved = list(example.resolved_pairs.values())[_id][0]
        fqn = example.answers[_id]
        unresolved = trim_fqn(resolved, fqn, node_type)
        joins.append(unresolved)

    blank_code_splits = example.blank_code.split('<blank>')
    assert len(blank_code_splits) == len(joins) + 1

    new_blank_code = ""
    for _id, code_split in enumerate(blank_code_splits[:-1]):
      new_blank_code += code_split + joins[_id]
    blank_code = new_blank_code + blank_code_splits[-1]

  else:
    answers = example.answers
    resolved_pairs = list(example.resolved_pairs.values())
    blank_code = example.blank_code

  if len(answers) == 0:
    return None

  code_lines, blank_code_lines = example.raw_code.split(";"), blank_code.split(";")
  glob_example, blank_ctr = [], 0
  for line_num, line in enumerate(blank_code_lines):
    blanks_idx = [m.start() for m in re.finditer('<blank>', line)]

    if len(blanks_idx) == 0:
      continue
    else:
      glob_example += process_line(line, line_num, code_lines, blank_code_lines,
                                   answers, resolved_pairs, blanks_idx, blank_ctr, context_length)
      blank_ctr += len(blanks_idx)
  return glob_example


def transform_cache(path_to_data, split, context_length):
  if split != 'test':
    data_path = Path(path_to_data) / f"examples_{split}.pkl"
    with open(data_path, 'rb') as file_obj:
      examples = pickle.load(file_obj)
  else:
    examples = get_test_examples(path_to_data)

  base_examples = []
  for example in tqdm(examples):
    glob_lines = transform_example(example, split, context_length)

    if glob_lines is None:
      continue

    base_examples.append(
      BaseInputExample(
        eid=example.eid, filename=example.filename,
        path_to_source=example.path_to_source, glob_lines=glob_lines,
      )
    )

  return base_examples


def load_examples(path_to_data, split, stage, context_length):
  try:
    data_path = Path(path_to_data) / f"examples_base_{split}_{stage}.pkl"
    with open(data_path, 'rb') as file_obj:
      base_examples = pickle.load(file_obj)

  except FileNotFoundError:
    base_examples = transform_cache(path_to_data, split, context_length)

    if split != 'test':
      project_examples = dict(zip(PROJECTS, [[] for _ in PROJECTS]))

      # Filter by projects
      for example in base_examples:
          for project in PROJECTS:
              if project in example.eid:
                  project_examples[project].append(example)

      assert len(base_examples) == sum([len(x) for x in list(project_examples.values())])

      base_examples_pt, base_examples_ft = [], []

      for project, p_examples in project_examples.items():
        num_examples_pt = int(len(p_examples) * 0.3)
        base_examples_pt += p_examples[:num_examples_pt]
        base_examples_ft += p_examples[num_examples_pt:]

      print(f'Number of pretraining examples: {len(base_examples_pt)}')
      print(f'Number of finetuning examples: {len(base_examples_ft)}')

      pt_data_path = Path(path_to_data) / f"examples_base_{split}_pt.pkl"
      with open(pt_data_path, 'wb') as file_obj:
        pickle.dump(base_examples_pt, file_obj)

      ft_data_path = Path(path_to_data) / f"examples_base_{split}_ft.pkl"
      with open(ft_data_path, 'wb') as file_obj:
        pickle.dump(base_examples_ft, file_obj)
    else:
      ft_data_path = Path(path_to_data) / f"examples_base_{split}_ft.pkl"
      with open(ft_data_path, 'wb') as file_obj:
        pickle.dump(base_examples, file_obj)

  return base_examples


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--path_to_data", required=True, type=str,
                      help="Path to prediction file.")
  args = parser.parse_args()

  # Test logic for transforming input examples.
  _test_transform_example()

  for split in ['train', 'valid', 'test']:
    for stage in ['pt', 'ft']:
      split_examples = load_examples(args.path_to_data, split, stage, CONTEXT_LENGTH)
