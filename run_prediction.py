import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

import utils
import bert_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="""The dataset on which to run predictions.""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="""Where to save the predictions.""")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="""Name of the dataset. Used for saving
                                predictions.""")
    parser.add_argument("--bert_weights_file", type=str, required=True,
                        help="""BERT weights checkpoint file.""")
    parser.add_argument("--bert_config_file", type=str, required=True,
                        help="""Config file for BERT model.""")
    parser.add_argument("--bert_model_class", type=str, required=True,
                        choices=["pooled", "entity"],
                        help="""BERT model class from bert_models.py
                                to use.""")
    parser.add_argument("--classes", type=str, nargs='*', default=[],
                        help="""List of unique classes the BERT model was
                                trained to predict.""")
    return parser.parse_args()


def main(args):
    dataset = pd.read_csv(args.dataset, index_col=0)

    lookup = {"pooled": bert_models.PooledModel,
              "entity": bert_models.EntityModel}
    bert_class = lookup[args.bert_model_class]

    bert = bert_class.from_model_checkpoint(
            args.bert_config_file, args.bert_weights_file)
    print(bert)

    if len(args.classes) > 0:
        unique_classes = sorted(args.classes)
        dataset = dataset[dataset["PREDICATE"].isin(unique_classes)]
    else:
        unique_classes = sorted(set(dataset["PREDICATE"].values))

    binarizer = LabelBinarizer()
    binarized_classes = binarizer.fit_transform(unique_classes)

    for (predicate, binarized) in zip(unique_classes, binarized_classes):
        print(f"{predicate:<16}: {binarized}")
    input()

    dataset_masked = utils.mask_dataset(dataset)
    print(dataset_masked.iloc[0]["SENTENCE"])
    dataset_masked.loc[:, "SENTENCE"] = dataset_masked["SENTENCE"].apply(
            lambda s: s + " [SEP]")
    print(dataset_masked.iloc[0]["SENTENCE"])
    input()
    predicates = dataset_masked["PREDICATE"].values
    y = binarizer.transform(predicates)
    assert dataset_masked.shape[0] == y.shape[0]

    os.makedirs(args.outdir, exist_ok=True)
    utils.evaluate(bert, dataset_masked, args.outdir, binarizer,
                   name=args.dataset_name, header=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
