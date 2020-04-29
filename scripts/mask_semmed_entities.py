import re
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import namedtuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semmed_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


def main(semmed_file, outdir):
    df = pd.read_csv(semmed_file, index_col=0)
    df.loc[:, "SENTENCE"] = df["SENTENCE"].apply(str.lower)

    Sentence = namedtuple("Sentence", ["sid", "pmid", "pred_ids"])

    sent_info = defaultdict(list)
    for (sentence, preds) in df.groupby("SENTENCE"):
        sentence = ' '.join(row.SENTENCE.lower().split())
        sent_id = hash(sentence)
        for pred in preds:
            pred_id = row.Index
            sent_info[sent_id].append(pred_id)

    # Get the gold-standard pairs
    sent2gold_pairs = get_gold_pairs(SI)

    # Get all subjects and objects
    sentences = [' '.join(sent.lower().split())
                 for (sent, _) in SI.groupby("SENTENCE")]
    all_subjs = set(SI["SUBJECT_TEXT"].str.lower().values)
    all_objs = set(SI["OBJECT_TEXT"].str.lower().values)

    annotated_sents = []
    predicates = []
    labels = []
    for (i, sentence) in enumerate(sentences):
        sent_id = f"S{i:07d}"
        pmid = pmid[sent_id]
        gold_pairs = sent2pairs[i]
        subj_obj2pred = {(s, o): p for (s, p, o) in gold_pairs}
        for subj in all_subjs:
            if subj not in sentence:
                continue
            for obj in all_objs:
                if obj not in sentence:
                    continue
                # TODO: replace_mentions(sentence, obj, subj) with label=0
                # maybe this will help it learn directionality.
                ann_sent = replace_mentions(sentence, subj, obj)
                if ann_sent is None:
                    continue
                if (subj, obj) in subj_obj2pred.keys():
                    label = 1
                    predicates.append(subj_obj2pred[(subj, obj)])
                else:
                    label = 0
                    predicates.append("NULL")
                annotated_sents.append(ann_sent)
                labels.append(label)

    annotated_sents = np.array(annotated_sents)
    labels = np.array(labels)


def get_gold_pairs(SI):
    sent2pairs = {}
    for (sent_id, (sent, preds)) in enumerate(SI.groupby("SENTENCE")):
        correct_pairs = set()
        for pred in preds.itertuples():
            subj = pred.SUBJECT_TEXT.lower()
            obj = pred.OBJECT_TEXT.lower()
            predicate = pred.PREDICATE
            label = pred.LABEL
            if label == 'y':
                correct_pairs.add((subj, predicate, obj))
        sent2pairs[sent_id] = correct_pairs
    return sent2pairs


def replace_mentions(sent, subj_text, obj_text):
    """
    Replace the subject of the predication with [SUBJ]
    and the object with [OBJ]. This is currently a hack because
    it relies on string matching instead of the indices provided
    in SemMedDB, as I haven't been able to reconcile those 
    indicies with the sentences.
    """
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


if __name__ == "__main__":
    args = parse_args()
    main(args.semmed_file, args.outdir)
