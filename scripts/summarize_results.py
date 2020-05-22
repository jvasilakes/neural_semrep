import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, \
                            classification_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prediction_files", type=str,
                        nargs='+', required=True)
    return parser.parse_args()


def main(args):
    prediction_files = [os.path.join(args.results_dir, f)
                        for f in args.prediction_files]
    print(prediction_files)

    metrics = []
    all_results = None
    for infile in prediction_files:
        results = pd.read_csv(infile, keep_default_na=False)
        all_results = pd.concat([all_results, results], axis=0)
        n_classes = results["GOLD"].unique().shape[0]
        if n_classes == 2:
            average = "binary"
        else:
            average = "weighted"

        gold = results["GOLD"].values
        predicted = results["PREDICTED"].values
        prec, rec, f, _ = precision_recall_fscore_support(
                gold, predicted, average=average)
        metrics.append([prec, rec, f])

    metrics = np.array(metrics)

    report = classification_report(all_results["GOLD"].values,
                                   all_results["PREDICTED"].values,
                                   digits=4)

    outfile = os.path.join(args.results_dir, "results_summary.txt")
    with open(outfile, 'w') as outF:
        outF.write("Averaged results over the files:\n")
        for f in prediction_files:
            outF.write(f + '\n')
        outF.write('\n')
        header = ["PRECISION", "RECALL", "F1"]
        outF.write(f"{header[0]:<21} {header[1]:<21} {header[2]:<21}\n")
        means = metrics.mean(axis=0)
        sds = metrics.std(axis=0)
        for (avg, sd) in zip(means, sds):
            outF.write(f"{avg:<7.4f} (+/- {sd:<7.4f}) ")
        outF.write('\n')
        outF.write(report)


if __name__ == "__main__":
    args = parse_args()
    main(args)
