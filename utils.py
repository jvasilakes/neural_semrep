import os
import re
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


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
                         "GOLD", "ARG2", "PREDICTED", "SCORES"])
        for (j, row) in enumerate(df.itertuples()):
            pid = row.PREDICATION_ID
            sent = row.SENTENCE
            subj = row.SUBJECT_TEXT
            gold = gold_labels[j]
            obj = row.OBJECT_TEXT
            pred = predicted_labels[j]
            score = ','.join([str(s) for s in scores[j]])
            writer.writerow([pid, sent, subj, gold, obj, pred, score])
    print(f"Predictions written to {predictions_outfile}")


def save_predictions(dataset, scores, outfile, labels2preds):
    prediction_ints = np.argmax(scores, axis=1)
    prediction_strings = [labels2preds[l] for l in prediction_ints]

    gold_labels = dataset["PREDICATE"]

    header = ["PREDICATION_ID", "SENTENCE", "GOLD", "PREDICTED"]
    score_cols = [f"score_{labels2preds[i]}"
                  for i in range(scores.shape[1])]
    header += score_cols

    with open(outfile, 'w') as outF:
        writer = csv.writer(outF)
        writer.writerow(header)
        for (i, row) in enumerate(dataset.itertuples()):
            pid = row.Index
            sent = row.SENTENCE
            gold = gold_labels[i]
            pred = prediction_strings[i]
            outrow = [pid, sent, gold, pred] + list(scores[i, :])
            writer.writerow(outrow)
    print(f"Predictions written to {outfile}")


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


def get_onehot_from_labels(intlabels):
    height = len(intlabels)
    width = max(intlabels) + 1
    onehots = np.zeros((height, width), dtype=int)
    for (i, l) in enumerate(intlabels):
        onehots[i][l] = 1
    return onehots


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
