import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="""CSV file of SemMedDB predications.""")
    parser.add_argument("--outfile", type=str, required=True,
                        help="""Where to save the augmented predications.""")
    parser.add_argument("--neg_label", default=0,
                        help="""The negative label.""")
    return parser.parse_args()


def main(infile, outfile, neg_label=0):
    orig = pd.read_csv(infile)

    # If the True/False annotation is False, make this a NULL predication.
    orig.loc[orig["LABEL"] == neg_label, "PREDICATE"] == "NULL"
    orig.loc[:, "PREDICATE"] = np.where(orig["LABEL"] == neg_label,
                                        "NULL",
                                        orig["PREDICATE"])
    num_null = (orig["PREDICATE"] == "NULL").sum()
    print(f"{num_null} predicates changed to 'NULL'.")
    orig.to_csv(outfile, header=True, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args.infile, args.outfile, neg_label=args.neg_label)
