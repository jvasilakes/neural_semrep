import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--keep_labels", type=str, nargs='*',
                        help="""Keep only examples with these labels.""")
    parser.add_argument("--no_split", action="store_true", default=False,
                        help="""Don't split dataset into train/val/test.""")
    return parser.parse_args()


def main(args):
    np.random.seed(args.random_seed)

    dataset = pd.read_csv(args.dataset, index_col=0)
    if args.keep_labels != []:
        print(dataset["PREDICATE"].unique())
        dataset = dataset[dataset["PREDICATE"].isin(args.keep_labels)]
        print(dataset["PREDICATE"].unique())

    # Change incorrect predications to have predicte NULL
    dataset.loc[:, "PREDICATE"] = np.where(dataset["LABEL"] == 'n',  # if
                                           "NULL",                   # then
                                           dataset["PREDICATE"])     # else

    # Mask out the subject and object of the predication in the sentence
    # with [ARG1] and [ARG2], respectively.
    dataset_masked, kept_indices = utils.mask_dataframe(dataset)

    # Normalize the spacing.
    dataset_masked.loc[:, "SENTENCE"] = dataset_masked["SENTENCE"].apply(
            lambda s: ' '.join(s.split()))

    # Binarize the labels
    predicates = dataset_masked["PREDICATE"].values
    if args.keep_labels != []:
        unique_classes = sorted(set(args.keep_labels))
    else:
        unique_classes = sorted(set(predicates))
    binarizer = LabelBinarizer()
    binarized_classes = binarizer.fit_transform(unique_classes)
    for (predicate, binarized) in zip(unique_classes, binarized_classes):
        print(f"{predicate:<16}: {binarized}")
    y = binarizer.transform(predicates)
    assert dataset_masked.shape[0] == y.shape[0]

    # Create a column for the binarized labels.
    # The number refers to the index of the 1 in the one-hot vector.
    int_labels = y.argmax(axis=1)
    dataset_masked["PREDICATE_LABEL"] = int_labels

    # Keep only the relevant columns, plus the Index (PREDICATION_ID)
    keep_cols = ["SENTENCE", "PREDICATE", "PREDICATE_LABEL"]
    dataset_masked = dataset_masked[keep_cols]

    if args.no_split is True:
        dataset_name = os.path.basename(args.dataset)
        outfile = os.path.join(args.outdir, dataset_name)
        dataset_masked.to_csv(outfile, header=True, index=True,
                              index_label="PREDICATION_ID")
        return

    # Split into train, validation, and test
    train_df, other_df, train_y, other_y = train_test_split(
            dataset_masked, y, train_size=0.8, stratify=y,
            random_state=args.random_seed)
    val_df, test_df, val_y, test_y = train_test_split(
            other_df, other_y, train_size=0.5, stratify=other_y,
            random_state=args.random_seed)
    assert train_df.shape[0] == train_y.shape[0]
    assert val_df.shape[0] == val_y.shape[0]
    assert test_df.shape[0] == test_y.shape[0]

    os.makedirs(args.outdir, exist_ok=False)
    train_file = os.path.join(args.outdir, "train.csv")
    val_file = os.path.join(args.outdir, "val.csv")
    test_file = os.path.join(args.outdir, "test.csv")
    train_df.to_csv(train_file, header=True, index=True,
                    index_label="PREDICATION_ID")
    val_df.to_csv(val_file, header=True, index=True,
                  index_label="PREDICATION_ID")
    test_df.to_csv(test_file, header=True, index=True,
                   index_label="PREDICATION_ID")


if __name__ == "__main__":
    args = parse_args()
    main(args)
