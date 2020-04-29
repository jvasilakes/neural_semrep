import os
import argparse
import re
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

import sys
sys.path.append("..")
from bert_model import BERTModel  # noqa: Module level import not at top of file.

np.random.seed(42)
tf.random.set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="""CSV dataset of labeled predications.""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="""Where to save the model predictions.""")
    parser.add_argument("--bert_model", type=str, required=True,
                        help="""tfhub.dev URL or file path to
                                saved BERT model.""")
    parser.add_argument("--nfolds", type=int, default=5,
                        help="""Number of CV folds. If 1, train on
                                the entire dataset.""")
    parser.add_argument("--mask_sentences", action="store_true", default=False,
                        help="""If True, mask subject and object mentions in
                                each sentence with [SUBJ] or [OBJ].""")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="""Where to save the model checkpoints
                                at each epoch.""")
    parser.add_argument("--tensorboard_logdir", type=str, default=None,
                        help="""Where to save the tensorboard logs.""")
    return parser.parse_args()


def main(dataset, outdir, bert_model, nfolds,
         mask_sentences, checkpoint_dir, tensorboard_logdir):
    df = pd.read_csv(dataset)

    keep_idxs = list(range(df.shape[0]))
    if mask_sentences is True:
        df, keep_idxs = mask_dataframe(df)

    predicates = df["PREDICATE"].values
    triples = get_predicate_triples(df)

    texts = [" [SEP] ".join([sent, triple]) for (sent, triple)
             in zip(df["SENTENCE"], triples)]
    texts = np.array(texts)

    y = (df["LABEL"].values == 'y').astype(int)
    print(y.shape)

    # Train on the entire dataset and save the model.
    if nfolds == 1:
        num_pos = y.sum()
        num_neg = y.shape[0] - num_pos
        initial_bias = compute_initial_bias(num_pos, num_neg)

        bert = BERTModel(n_classes=1, bert_file=bert_model,
                         max_seq_length=128, batch_size=16,
                         fit_metrics=["precision", "recall"],
                         initial_bias=initial_bias,
                         checkpoint_dir=checkpoint_dir,
                         logdir=tensorboard_logdir)
        bert.fit(texts, y, epochs=4)
        return

    precs = []
    recs = []
    f1s = []
    kf = StratifiedKFold(n_splits=nfolds, shuffle=True)
    for (i, (train_idxs, test_idxs)) in enumerate(kf.split(texts, predicates)):
        print(f"=========")
        print(f"Fold: {i + 1}")
        print(f"=========")
        train_texts, test_texts = texts[train_idxs], texts[test_idxs]
        y_train, y_test = y[train_idxs], y[test_idxs]

        num_pos = y_train.sum()
        num_neg = y_train.shape[0] - num_pos
        initial_bias = compute_initial_bias(num_pos, num_neg)

        logdir = os.path.join(tensorboard_logdir, str(i))
        os.makedirs(logdir, exist_ok=False)
        bert = BERTModel(n_classes=1, bert_file=bert_model,
                         max_seq_length=128, batch_size=16,
                         fit_metrics=["precision", "recall"],
                         initial_bias=initial_bias,
                         checkpoint_dir=checkpoint_dir,
                         logdir=logdir)

        initial_loss = bert.compute_loss(train_texts, y_train)[0]
        expected_loss = compute_expected_loss(num_pos, y_train.shape[0])
        print("------------------------------")
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Expected loss: {expected_loss:.4f}")
        print("------------------------------")

        bert.fit(train_texts, y_train, epochs=4)
        preds = bert.predict(test_texts, predict_classes=True)
        p, r, f, _ = precision_recall_fscore_support(y_test, preds,
                                                     average="binary")
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")

        test_df = df.iloc[test_idxs]
        scores = bert.predict(test_texts, predict_classes=False)
        predictions_outfile = os.path.join(
                outdir, f"predictions_fold{i + 1}.csv")
        with open(predictions_outfile, 'w') as outF:
            writer = csv.writer(outF)
            writer.writerow(["PREDICATION_ID", "SENTENCE", "SUBJ",
                             "PREDICATE", "OBJ", "PREDICTED",
                             "GOLD", "SCORE_x=1"])
            for (j, row) in enumerate(test_df.itertuples()):
                pid = row.PREDICATION_ID
                sent = row.SENTENCE
                subj = row.SUBJECT_TEXT
                predicate = row.PREDICATE
                obj = row.OBJECT_TEXT
                pred = int(preds[j])
                gold = y_test[j]
                score = float(scores[j])
                writer.writerow([pid, sent, subj, predicate,
                                 obj, pred, gold, score])
        metrics_outfile = os.path.join(outdir, f"metrics.csv")
        mode = 'a'
        if i == 0:
            mode = 'w'
        with open(metrics_outfile, mode) as outF:
            writer = csv.writer(outF)
            if i == 0:
                writer.writerow(["FOLD", "PRECISION", "RECALL", "F1"])
            writer.writerow([i + 1, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"])
        print(f"Predictions written to {predictions_outfile}")
        print(f"Performance metrics written to {metrics_outfile}")

        tf.keras.backend.clear_session()

    print()
    print(f"Precision: {np.mean(precs):.4f} (+/- {np.std(precs):.4f})")
    print(precs)
    print()
    print(f"Recall: {np.mean(recs):.4f} (+/- {np.std(recs):.4f})")
    print(recs)
    print()
    print(f"F1 score: {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
    print(f1s)


def mask_dataframe(dataframe):
    masked_rows = []
    keep_idxs = []
    for (i, row) in enumerate(dataframe.itertuples()):
        sent = row.SENTENCE
        subj = row.SUBJECT_TEXT
        obj = row.OBJECT_TEXT
        masked = mask_mentions(sent, subj, obj)
        if masked is not None:
            keep_idxs.append(i)
            new_row = row._asdict()
            new_row["SUBJECT_TEXT"] = "[SUBJ]"
            new_row["OBJECT_TEXT"] = "[OBJ]"
            new_row["SENTENCE"] = masked
            masked_rows.append(new_row)
    masked_df = pd.DataFrame(masked_rows)
    return masked_df, keep_idxs


def mask_mentions(sent, subj_text, obj_text):
    if subj_text == obj_text:
        return None

    subj_text_esc = re.escape(subj_text)
    if len(re.findall(fr"\b{subj_text_esc}\b", sent)) > 1:
        return None
    try:
        subj_start, subj_end = re.search(fr"\b{subj_text_esc}\b", sent).span()
    except AttributeError:
        return None
    new_sent = sent[:subj_start] + "[SUBJ]" + sent[subj_end:]

    obj_text_esc = re.escape(obj_text)
    if len(re.findall(fr"\b{subj_text_esc}\b", sent)) > 1:
        return None
    try:
        obj_start, obj_end = re.search(fr"\b{obj_text_esc}\b", new_sent).span()
    except AttributeError:
        return None
    new_sent = new_sent[:obj_start] + "[OBJ]" + new_sent[obj_end:]
    return new_sent


def get_predicate_triples(dataframe):
    subjs = dataframe["SUBJECT_TEXT"]
    predicates = dataframe["PREDICATE"]
    objs = dataframe["OBJECT_TEXT"]
    triples = np.array([' '.join(elems) for elems
                        in zip(subjs, predicates, objs)])
    return triples


def compute_initial_bias(num_pos, num_neg):
    return np.log([num_pos / num_neg])


def compute_expected_loss(num_pos, total):
    p_0 = num_pos / total
    return -p_0 * np.log(p_0) - (1 - p_0) * np.log(1-p_0)


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset, args.outdir, args.bert_model, args.nfolds,
         args.mask_sentences, args.checkpoint_dir, args.tensorboard_logdir)
