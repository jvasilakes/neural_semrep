import os
import csv
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from collections import defaultdict, Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_files", type=str, nargs='+')
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--neg_label", type=str, default="NULL")
    return parser.parse_args()


def main(args):

    os.makedirs(args.outdir, exist_ok=True)

    # Performance metrics for each prediction file will be saved here.
    ind_fname = f"file_metrics_{args.dataset}.csv"
    ind_metrics_outfile = os.path.join(args.outdir, ind_fname)
    if os.path.exists(ind_metrics_outfile):
        raise OSError(f"Existing metrics file found in {args.outdir}.")

    # Averaged performance metrics across prediction files will be saved here.
    avg_fname = f"averaged_metrics_{args.dataset}.csv"
    avg_metrics_outfile = os.path.join(args.outdir, avg_fname)
    if os.path.exists(avg_metrics_outfile):
        raise OSError(f"Existing metrics file found in {args.outdir}.")

    metrics_to_average = defaultdict(list)
    for (i, infile) in enumerate(args.prediction_files):
        results = pd.read_csv(infile, keep_default_na=False)

        gold_labels = results["GOLD"].values
        predictions = results["PREDICTED"].values
        num_predicted = Counter(predictions)
        assert len(set(gold_labels)) > 1

        report = classification_report(gold_labels, predictions,
                                       output_dict=True, zero_division=0)
        metric_names = ["precision", "recall", "f1-score", "support"]
        with open(ind_metrics_outfile, 'a') as outF:
            writer = csv.writer(outF)
            if i == 0:
                writer.writerow(["file", "label", "precision", "recall",
                                 "f1", "num_gold", "num_predicted"])
            for key in report.keys():
                if key == "accuracy":
                    continue
                metrics_list = [report[key][metric] for metric in metric_names]
                if key in ["macro avg", "weighted avg"]:
                    metrics_list += [len(predictions)]
                else:
                    metrics_list += [num_predicted[key]]
                metrics_list_format = [f"{m:.4f}" if i < 3 else str(m)
                                       for (i, m) in enumerate(metrics_list)]
                writer.writerow([infile, key] + metrics_list_format)
                # Save these metrics to average over later.
                metrics_to_average[key].append(metrics_list)

    # Average over the prediction_files
    with open(avg_metrics_outfile, 'w') as outF:
        writer = csv.writer(outF)
        writer.writerow(["label",
                         "precision_mean", "precision_sd",
                         "recall_mean", "recall_sd",
                         "f1_mean", "f1_sd",
                         "num_gold_mean", "num_gold_sd",
                         "num_predicted_mean", "num_predicted_sd"])
        for (key, nums) in metrics_to_average.items():
            avgs = np.mean(nums, axis=0)
            avgs = [f"{a:.4f}" for a in avgs]
            sds = np.std(nums, axis=0)
            sds = [f"{s:.4f}" for s in sds]
            zipped = [x for pair in zip(avgs, sds) for x in pair]
            writer.writerow([key] + zipped)


if __name__ == "__main__":
    args = parse_args()
    main(args)
