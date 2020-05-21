import os
import re
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def mask_dataset(dataset):
    masked_rows = []
    for (i, row) in enumerate(dataset.itertuples()):
        sentence = row.SENTENCE
        arg1 = row.SUBJECT_TEXT
        arg2 = row.OBJECT_TEXT
        masked = mask_mentions(sentence, arg1, arg2)
        if masked is not None:
            new_row = row._asdict()
            new_row["SENTENCE"] = masked
            masked_rows.append(new_row)
    masked_dataset = pd.DataFrame(masked_rows)
    masked_dataset = masked_dataset.set_index("Index")
    return masked_dataset


def mask_mentions(sentence, arg1_text, arg2_text):
    if arg1_text == arg2_text:
        return None

    arg1_esc = re.escape(arg1_text)
    # There are multiple mentions of ARG1.
    if len(re.findall(fr"\b{arg1_esc}\b", sentence)) > 1:
        return None
    try:
        start1, end1 = re.search(fr"\b{arg1_esc}\b", sentence).span()
    except AttributeError:  # NoneType
        return None
    masked = sentence[:start1] + "[ARG1]" + sentence[end1:]

    # There are multiple mentions of ARG2.
    arg2_esc = re.escape(arg2_text)
    if len(re.findall(fr"\b{arg2_esc}\b", sentence)) > 1:
        return None
    try:
        start2, end2 = re.search(fr"\b{arg2_esc}\b", masked).span()
    except AttributeError:  # NoneType
        return None
    masked = masked[:start2] + "[ARG2]" + masked[end2:]
    return masked


def compute_initial_bias(y):
    """Assumes y is binary encoded, e.g. LabelBinarizer()"""
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    biases = np.log(pos / neg)
    return biases


def evaluate(bert_model, dataset, outdir, label_binarizer,
             name="dataset", header=True):
    docs = dataset["SENTENCE"].values
    labels = dataset["PREDICATE"].values
    predictions = bert_model.predict(docs, predict_classes=True)
    predicted_labels = label_binarizer.inverse_transform(predictions)
    p, r, f, _ = precision_recall_fscore_support(
            labels, predicted_labels, average="weighted")
    print("------------------------------------------------------------")
    print(f"Dataset: {name}")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f} (weighted)")
    print("------------------------------------------------------------")

    scores = bert_model.predict(docs, predict_classes=False)
    predictions_outfile = os.path.join(outdir, f"predictions_{name}.csv")
    with open(predictions_outfile, 'w') as outF:
        writer = csv.writer(outF)
        writer.writerow(["PREDICATION_ID", "SENTENCE", "ARG1",
                         "ARG2", "GOLD", "PREDICTED", "SCORES"])
        for (i, row) in enumerate(dataset.itertuples()):
            pid = row.Index
            sent = row.SENTENCE
            arg1 = row.SUBJECT_TEXT
            arg2 = row.OBJECT_TEXT
            gold = row.PREDICATE
            prediction = predicted_labels[i]
            score = ','.join([str(s) for s in scores[i]])
            writer.writerow([pid, sent, arg1, arg2, gold, prediction, score])
    print(f"{name} predictions for written to '{predictions_outfile}'")
