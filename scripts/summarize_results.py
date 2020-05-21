import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import precision_recall_curve, average_precision_score, \
                            precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="""Where the predictions for each
                                fold were saved.""")
    parser.add_argument("--plot_title", type=str, default=None,
                        help="""What to title the plot.""")
    parser.add_argument("--save", action="store_true", default=False,
                        help="""If specified, save plot and
                                results to results_dir.""")
    parser.add_argument("--prediction_files", type=str, nargs='*',
                        default=[])
    return parser.parse_args()


def main(results_dir, plot_title=None, save=False, prediction_files=[]):
    if len(prediction_files) == 0:
        prediction_files = glob(os.path.join(results_dir,
                                             "predictions_*.csv"))
    else:
        prediction_files = [os.path.join(results_dir, f)
                            for f in prediction_files]
    all_results = pd.DataFrame()
    for infile in prediction_files:
        fold_results = pd.read_csv(infile, keep_default_na=False)
        all_results = pd.concat((all_results, fold_results))
    print(f"N={all_results.shape[0]}\n")
    n_classes = all_results["GOLD_PREDICATION"].unique().shape[0]
    if n_classes == 2:
        average = "binary"
    else:
        average = "weighted"

    metrics_outfile = None
    if save is True:
        metrics_outfile = os.path.join(results_dir, "metrics_by_predicate.csv")
    metrics_by_predicate(all_results, outfile=metrics_outfile, average=average)

    if n_classes == 2:
        plot_outfile = None
        if save is True:
            plot_outfile = os.path.join(results_dir, "precision_recall_curve.png")
        plot_precision_recall_curve(all_results, plot_title=plot_title,
                                    outfile=plot_outfile)


def metrics_by_predicate(results_df, outfile=None, average="binary"):
    total_p, total_r, total_f, _ = precision_recall_fscore_support(
            results_df["GOLD"].values, results_df["PREDICTED"].values,
            average=average, zero_division=0)
    total_row = {"PREDICATE": "ALL", "N": results_df.shape[0],
                 "PRECISION": total_p, "RECALL": total_r, "F1": total_f}
    metrics_rows = [total_row]
    for (predicate, group) in results_df.groupby("PREDICATE"):
        prec, rec, f1, _ = precision_recall_fscore_support(
                group["GOLD"].values, group["PREDICTED"].values,
                average=average, zero_division=0)
        row = {"PREDICATE": predicate, "N": group.shape[0],
               "PRECISION": prec, "RECALL": rec, "F1": f1}
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    if outfile is not None:
        metrics_df.to_csv(outfile, header=True, index=False)
    else:
        print(metrics_df)


def plot_precision_recall_curve(results_df, plot_title=None, outfile=None):
    y_true = results_df["GOLD"].values
    y_score = results_df["SCORES"].values
    precs, recs, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.step(recs, precs, color='b', where='post',
            label=f"All Predicates (N={y_true.shape[0]})")

    for (pred, group) in results_df.groupby("PREDICATE"):
        pred_y_true = group["GOLD"].values
        pred_y_score = group["SCORE_x=1"].values
        pred_precs, pred_recs, _ = precision_recall_curve(
                pred_y_true, pred_y_score)
        label = f"{pred} (N={pred_y_true.shape[0]})"
        ax.step(pred_recs, pred_precs, alpha=0.4, where='post', label=label)

    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")

    box_props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    avg_prec = average_precision_score(y_true, y_score)
    avg_text = f"Average Precision\n{avg_prec:.4f}"
    ax.text(0.7, 0.15, avg_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=box_props)

    if plot_title is not None:
        plt.title(plot_title, fontsize=18)

    if outfile is not None:
        plt.savefig(outfile)
    else:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.results_dir, plot_title=args.plot_title, save=args.save,
         prediction_files=args.prediction_files)
