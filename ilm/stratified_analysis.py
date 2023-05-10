import pickle
import statistics
from collections import Counter

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from transformers import GPT2Tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str,
                        help="Path to prediction pickle file.")
    args = parser.parse_args()

    with open(args.path, 'rb') as f:
        test_preds = pickle.load(f)

    stratified_items = {}
    for strata in [1, 2, 3, 4, 5, 10, 15]:
        stratified_items[strata] = {'True': [], 'Pred': []}


    for item in test_preds:
        item_preds = item['Prediction'].split('<|end-of-infill|>')[:-1]
        item_true = item['True']
        item_true = item_true.replace('<|start-of-infill|>', '')
        item_true = item_true.split('<|end-of-infill|>')[:-1]

        if len(item_preds) in [1, 2, 3, 4]:
            strata_key = len(item_preds)
        elif 5 <= len(item_preds) < 10:
            strata_key = 5
        elif 10 <= len(item_preds) < 15:
            strata_key = 10
        elif 15 <= len(item_preds) < 20:
            strata_key = 15

        strata_true = stratified_items[strata_key]['True']
        strata_true += item_true
        strata_pred = stratified_items[strata_key]['Pred']
        strata_pred += item_preds

        stratified_items[strata_key]['True'] = strata_true
        stratified_items[strata_key]['Pred'] = strata_pred

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    for k, v in stratified_items.items():
        EM = sum([1 for x, y in zip(v['True'], v['Pred']) if x == y]) / len(v['True'])
        rouge_scores = [rscorer.score(pred, v['True'][i]) for i, pred in enumerate(v['Pred'])]
        avg_rouge = statistics.mean([x['rougeL'].fmeasure for x in rouge_scores])

        bleu_score = corpus_bleu([[tokenizer.tokenize(x)] for x in v['True']],
                                 [tokenizer.tokenize(y) for y in v['Pred']],
                                 weights=(0.5, 0.5))
        print(f"{k}\t{EM}\t{avg_rouge}\t{bleu_score}\tTotal: {len(v['True'])}")
