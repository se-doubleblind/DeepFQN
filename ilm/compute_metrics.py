import json
import argparse
import statistics
from collections import Counter


from rouge_score import rouge_scorer
import ilm.tokenize_util
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def compute_eval_metrics(items, tokenizer):
    '''Calculates evaluation metrics such as Exact Match (EM) Accuracy,
    ROUGE-L, and BLEU-2 scores.

    Arguments:
        items (list of dict):
            Dictionary items containing 'True' and 'Prediction' values for 
            a given FQN.
        tokenizer (transformers.GPT2Tokenizer):
            GPT-2 tokenizer.
    '''
    total, tp = 0, 0
    correct = {}
    incorrect = []
    rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_scores, bleu2_scores, bleu4_scores = [], [], []
    all_true_toks, all_pred_toks = [], []

    for item in items:
        try:
            true, preds = item['True'], item['Prediction']
            true = true.replace('<|start-of-infill|>', '')
            true = true.split('<|end-of-infill|>')[:-1]
            preds = preds.replace('<|start-of-infill|>', '')
            preds = preds.split('<|end-of-infill|>')[:-1]

            if len(preds) < len(true):
                true = true[:len(preds)]
            else:
                preds = preds[:len(true)]

            total += len(preds)
            for i, pred in enumerate(preds):
                if pred == true[i]:
                    tp += 1
                    if pred in correct:
                        pred_correct = correct[pred]
                        correct[pred] = pred_correct + 1
                    else:
                        correct[pred] = 1
                else:
                    incorrect.append((pred, true[i]))

                pred_toks = ilm.tokenize_util.tokenize(pred, tokenizer)
                true_toks = ilm.tokenize_util.tokenize(true[i], tokenizer)
                all_pred_toks.append(pred_toks)
                all_true_toks.append([true_toks])

                rouge_scores.append(rscorer.score(pred, true[i]))
        except Exception as e:
            print(e)
            pass

    correct = sorted(correct.items(), key=lambda x:x[1], reverse=True)

    print(f'Exact Match Accuracy: {tp/total}')
    print(f'Correctly Predicted: {tp}\t Total: {total}')
    print()
    print(f"Mean ROUGE-LCS Precision: {statistics.mean([score['rougeL'].precision for score in rouge_scores])}")
    print(f"Mean ROUGE-LCS Recall: {statistics.mean([score['rougeL'].recall for score in rouge_scores])}")
    print(f"Mean ROUGE-LCS F-Score: {statistics.mean([score['rougeL'].fmeasure for score in rouge_scores])}")
    print()
    print(f"Corpus BLEU-2 Score: {corpus_bleu(all_true_toks, all_pred_toks, weights=(0.5, 0.5))}")
    print()

    incorrect_counter_top10 = Counter([x[1] for x in incorrect]).most_common(10)
    print('Printing Top-10 FQNs with incorrect predictions')
    print(incorrect_counter_top10)
    print()
    print('Printing Top-10 FQN <prediction, true> tuples')
    print(sorted(incorrect, key=lambda x: x[1])[:10])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str,
                        help="Path to prediction file.")
    args = parser.parse_args()

    with open(args.path, 'r') as fileobj:
        predictions = json.load(fileobj)

    compute_eval_metrics(predictions)