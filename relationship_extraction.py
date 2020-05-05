import os
import re
import csv
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter

import sys
sys.path.append("..")
import bert_model  # noqa: Module level import not at top of file.

np.random.seed(42)
tf.random.set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="""CSV dataset of labeled predications.""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="""Where to save the model predictions.""")
    parser.add_argument("--bert_model_file", type=str, required=True,
                        help="""tfhub.dev URL or file path to
                                saved BERT model.""")
    parser.add_argument("--bert_model_class", type=str, default="pooled",
                        choices=["pooled", "entity"],
                        help="""Which model class (from bert_models.py)
                                to use.""")
    parser.add_argument("--mask_sentences", action="store_true", default=False,
                        help="""If True, mask subject and object mentions in
                                each sentence with [ARG1] or [ARG2].""")
    parser.add_argument("--epochs", type=int, default=1,
                        help="""Number of training epochs.""")
    parser.add_argument("--checkpoint_dirname", type=str, default=None,
                        help="""Where to save the model checkpoints
                                at each epoch.""")
    parser.add_argument("--tensorboard_dirname", type=str, default=None,
                        help="""Where to save the tensorboard logs.""")
    parser.add_argument("--from_model_checkpoint", type=str, default=None,
                        help="""Initialize from a previous checkpoint.""")
    return parser.parse_args()


def main(args):
    df = pd.read_csv(args.dataset)

    keep_idxs = list(range(df.shape[0]))
    if args.mask_sentences is True:
        df, keep_idxs = mask_dataframe(df)
    print(df.shape)

    # Assign a NULL predicate to any false predications.
    df.loc[:, "PREDICATE"] = np.where(df["LABEL"] == 'n',  # if
                                      "NULL",              # then
                                      df["PREDICATE"])     # else

    predicates = df["PREDICATE"].values
    binarizer = LabelBinarizer()
    y = binarizer.fit_transform(predicates)
    n_classes = len(set(predicates))
    unique_predicates = list(set(predicates))
    print("LABELS: ", unique_predicates)
    print(binarizer.transform(unique_predicates))

    # BERT requires [SEP] at the end.
    texts = df["SENTENCE"].apply(lambda s: s + " [SEP]").values

    # 80% train, 10% validation, 10% test
    train_splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8,
                                            random_state=42)
    train_idxs, other_idxs = next(train_splitter.split(texts, y))
    train_texts, other_texts = texts[train_idxs], texts[other_idxs]
    train_y, other_y = y[train_idxs], y[other_idxs]

    test_splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.5,
                                           random_state=42)
    val_idxs, test_idxs = next(test_splitter.split(other_texts, other_y))
    val_texts, test_texts = other_texts[val_idxs], other_texts[test_idxs]
    val_y, test_y = other_y[val_idxs], other_y[test_idxs]

    train_counts = Counter(binarizer.inverse_transform(train_y))
    val_counts = Counter(binarizer.inverse_transform(val_y))
    test_counts = Counter(binarizer.inverse_transform(test_y))

    print("TRAIN: ", train_texts.shape[0], train_counts)
    print("VAL: ", val_texts.shape[0], val_counts)
    print("TEST: ", test_texts.shape[0], test_counts)

    initial_bias = compute_initial_bias(train_y)
    print(f"Initial bias: {initial_bias}")

    bert_class = get_bert_model_class(args.bert_model_class)
    print("Using BERT model: ", bert_class)
    input()

    os.makedirs(args.outdir, exist_ok=False)
    ckpt_dir = tb_logdir = None
    if args.checkpoint_dirname is not None:
        ckpt_dir = os.path.join(args.outdir, args.checkpoint_dirname)
        os.makedirs(ckpt_dir, exist_ok=True)
    if args.tensorboard_dirname is not None:
        tb_logdir = os.path.join(args.outdir, args.tensorboard_dirname)
        os.makedirs(tb_logdir, exist_ok=True)
    if args.from_model_checkpoint is not None:
        weights_file = args.from_model_checkpoint
        params_file = os.path.join(os.path.dirname(weights_file),
                                   f"../{bert_class.name}.json")
        bert = bert_class.from_model_checkpoint(params_file, weights_file)
    else:
        bert = bert_class(n_classes=n_classes, bert_file=args.bert_model_file,
                          max_seq_length=256, batch_size=16,
                          dropout_rate=0.2, learning_rate=2e-5,
                          initial_bias=initial_bias,
                          fit_metrics=["precision", "recall"],
                          validation_data=(val_texts, val_y),
                          checkpoint_dir=ckpt_dir,
                          logdir=tb_logdir)
    bert.save_params(os.path.join(args.outdir, f"{bert_class.name}.json"))

    initial_loss = bert.compute_loss(train_texts, train_y)[0]
    if n_classes == 1:
        expected_loss = compute_expected_loss_binary(y)
    else:
        expected_loss = compute_expected_loss_categorical(y)
    print("------------------------------")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Expected loss: {expected_loss:.4f}")
    print("------------------------------")

    # TRAIN
    # Validate on validation_data at each epoch
    bert.fit(train_texts, train_y, epochs=args.epochs)

    # EVALUATE
    # Save test examples with their prediction and their scores.
    test_df = df.iloc[test_idxs]
    evaluate(bert, test_df, test_texts, test_y, args.outdir,
             binarizer, name="test", header=True)
    val_df = df.iloc[val_idxs]
    evaluate(bert, val_df, val_texts, val_y, args.outdir,
             binarizer, name="val", header=False)
    train_df = df.iloc[train_idxs]
    evaluate(bert, train_df, train_texts, train_y, args.outdir,
             binarizer, name="train", header=False)


def evaluate(bert_model, df, texts, y, outdir,
             label_binarizer, name="test", header=True):
    predictions = bert_model.predict(texts, predict_classes=True).numpy()
    p, r, f, _ = precision_recall_fscore_support(y, predictions,
                                                 average="micro")
    print("---------------------------------------------------------")
    print(f"Dataset: {name}")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f} (micro)")
    print("---------------------------------------------------------\n")
    # Save evaluation metrics
    metrics_outfile = os.path.join(outdir, f"metrics.csv")
    with open(metrics_outfile, 'a') as outF:
        writer = csv.writer(outF)
        if header is True:
            writer.writerow(["DATASET", "PRECISION", "RECALL", "F1"])
        writer.writerow([name, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"])

    scores = bert_model.predict(texts, predict_classes=False).numpy()
    predictions_outfile = os.path.join(
            outdir, f"predictions_{name}.csv")
    predicted_labels = label_binarizer.inverse_transform(predictions)
    gold_labels = label_binarizer.inverse_transform(y)
    with open(predictions_outfile, 'w') as outF:
        writer = csv.writer(outF)
        writer.writerow(["PREDICATION_ID", "SENTENCE", "ARG1",
                         "PREDICATE", "ARG2", "PREDICTED",
                         "GOLD", "SCORES"])
        for (j, row) in enumerate(df.itertuples()):
            pid = row.PREDICATION_ID
            sent = row.SENTENCE
            subj = row.SUBJECT_TEXT
            predicate = gold_labels[j]
            obj = row.OBJECT_TEXT
            pred = predicted_labels[j]
            score = ','.join([str(s) for s in scores[j]])
            writer.writerow([pid, sent, subj, predicate,
                             obj, pred, gold, score])
    print(f"Predictions written to {predictions_outfile}")


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
            new_row["SUBJECT_TEXT"] = "[ARG1]"
            new_row["OBJECT_TEXT"] = "[ARG2]"
            new_row["SENTENCE"] = masked
            masked_rows.append(new_row)
    masked_df = pd.DataFrame(masked_rows)
    masked_df = masked_df.set_index("Index")
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
    new_sent = sent[:subj_start] + "[ARG1]" + sent[subj_end:]

    obj_text_esc = re.escape(obj_text)
    if len(re.findall(fr"\b{subj_text_esc}\b", sent)) > 1:
        return None
    try:
        obj_start, obj_end = re.search(fr"\b{obj_text_esc}\b", new_sent).span()
    except AttributeError:
        return None
    new_sent = new_sent[:obj_start] + "[ARG2]" + new_sent[obj_end:]
    return new_sent


def get_predicate_triples(dataframe):
    subjs = dataframe["SUBJECT_TEXT"]
    predicates = dataframe["PREDICATE"]
    objs = dataframe["OBJECT_TEXT"]
    triples = np.array([' '.join(elems) for elems
                        in zip(subjs, predicates, objs)])
    return triples


def compute_initial_bias(labels):
    """Assumes labels are binary encoded"""
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    biases = np.log(pos / neg)
    return biases


def compute_expected_loss_binary(labels):
    """Assumes 1 is the positive label and 0 the negative label."""
    p_0 = labels.sum() / labels.shape[0]
    return -p_0 * np.log(p_0) - (1 - p_0) * np.log(1-p_0)


def compute_expected_loss_categorical(labels):
    """Assumes labels are binary encoded"""
    p_0 = labels.sum(axis=0) / labels.shape[0]
    return -np.sum(p_0 * np.log(p_0))


def get_bert_model_class(model_class_string):
    lookup = {
              "pooled": bert_model.PooledModel,
              "entity": bert_model.EntityModel,
              }
    try:
        return lookup[model_class_string]
    except KeyError:
        raise KeyError(f"BERT model '{model_class_string}' not supported.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
