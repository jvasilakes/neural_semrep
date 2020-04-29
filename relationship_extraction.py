import os
import csv
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from bert_model import BERTModel


BERT_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semmeddb_splits_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bert_file", type=str, default=BERT_URL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--test_init_predictions", action="store_true",
                        default=False)
    parser.add_argument("--save_predictions_to", type=str, default=None)
    return parser.parse_args()


def main(semmeddb_dir, max_seq_length, batch_size, bert_file, epochs,
         outdir=None, test_init_predictions=False):
    train_df = pd.read_csv(f"{semmeddb_dir}/train.csv")
    test_df = pd.read_csv(f"{semmeddb_dir}/test.csv")
    val_df = pd.read_csv(f"{semmeddb_dir}/validation.csv")

    train_sents, train_y = train_df["SENTENCE"].values, train_df["LABEL"].values  # noqa
    test_sents, test_y = test_df["SENTENCE"].values, test_df["LABEL"].values
    val_sents, val_y = val_df["SENTENCE"].values, val_df["LABEL"].values

    total_examples = len(train_y)
    num_pos = train_y.sum()
    num_neg = total_examples - num_pos
    # initial_bias = None
    initial_bias = compute_initial_bias(num_pos, num_neg)

    bert_model = BERTModel(bert_file=bert_file,
                           max_seq_length=max_seq_length,
                           batch_size=batch_size,
                           initial_bias=initial_bias,
                           validation_data=(val_sents[:10], val_y[:10]),
                           fit_metrics=["precision", "recall"])

    print(bert_model.model.summary())
    print(f"Total: {total_examples} (+: {num_pos}, -: {num_neg})")
    print("initial bias=", initial_bias)
    if test_init_predictions is True:
        print("Initial Predictions")
        print("-------------------")
        scores = bert_model.predict(test_sents[:10], predict_classes=False)
        loss = bert_model.compute_loss(train_sents, train_y)[0]
        print("Scores")
        print(scores)
        print("Expected score: ", num_pos / total_examples)
        print(f"Train loss {loss:.4f}")
        print(f"Expected loss {expected_loss(num_pos, total_examples):.4f}")
        print("-------------------")

    bert_model.fit(train_sents[:100], train_y[:100], epochs=epochs)

    val_outfile = test_outfile = None
    if outdir is not None:
        os.makedirs(outdir)
        val_outfile = os.path.join(outdir, "val_predictions.csv")
        test_outfile = os.path.join(outdir, "test_predictions.csv")
    evaluate(bert_model, val_sents[:10], val_y[:10], outfile=val_outfile)
    evaluate(bert_model, test_sents[:10], test_y[:10], outfile=test_outfile)


def evaluate(model, test_sents, test_y, outfile=None):
    predictions = model.predict(test_sents, predict_classes=True)
    prec, rec, f1, _ = precision_recall_fscore_support(test_y, predictions,
                                                       average="binary",
                                                       zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

    if outfile is not None:
        with open(outfile, 'w') as outF:
            writer = csv.writer(outF, delimiter=',')
            for (sent, gold, pred) in zip(test_sents, test_y, predictions):
                writer.writerow([sent, gold, pred])


def compute_initial_bias(num_pos, num_neg):
    return np.log([num_pos / num_neg])


def expected_loss(num_pos, total):
    p_0 = num_pos / total
    return -p_0 * np.log(p_0) - (1 - p_0) * np.log(1-p_0)


if __name__ == "__main__":
    args = parse_args()
    main(args.semmeddb_splits_dir, args.max_seq_length,
         args.batch_size, args.bert_file, args.epochs,
         test_init_predictions=args.test_init_predictions,
         outdir=args.save_predictions_to)
