import json
import shutil
import logging
import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

import torch


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECTS = ['android', 'gwt', 'hibernate-orm', 'jdk', 'joda-time', 'xstream']
#PROJECTS = ['hibernate-orm']

def train_bpe_tokenizer(vocab_size):
    code_examples = []
    for project in PROJECTS:
        data_path = Path('../../full-fqn') / f'{project}.json'

        with open(str(data_path), 'r', encoding='utf-8') as file_obj:
            project_examples = json.load(file_obj)

        code_examples += [example['code'] for example in project_examples]

    # BPE tokenizer accepts inputs as a set of files.
    tmp_data_path = Path('tmp')
    tmp_data_path.mkdir(exist_ok=True, parents=True)

    files = []
    for _id, code in enumerate(code_examples):
        tmp_file_path = str(tmp_data_path / f'tmp{_id}.txt')
        with open(tmp_file_path, 'w', encoding='utf-8') as file_obj:
            file_obj.write(code)
        files.append(tmp_file_path)

    path_to_tok = Path(f'custom_bpe_encoder')
    path_to_tok.mkdir(exist_ok=True, parents=True)

    logger.info(f"Training BPE tokenizer for predicting FQNs.")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=files,
                    vocab_size=vocab_size,
                    min_frequency=2)
    tokenizer.save_model(str(path_to_tok))
    shutil.rmtree(str(tmp_data_path))
    logger.info(f"Saved BPE tokenizer in /custom_bpe_encoder.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tok", action='store_true',
                        help="Train byte-level BPE tokenizer")
    parser.add_argument("--vocab_size", default=8, type=int,
                        help="Vocabulary size")
    args = parser.parse_args()

    if args.train_tok:
        train_bpe_tokenizer(vocab_size=args.vocab_size)
