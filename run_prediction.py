import argparse
import pandas as pd

import utils
import bert_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="""The dataset on which to run predictions.""")
    parser.add_argument("--outfile", type=str, required=True,
                        help="""Where to save the predictions.""")
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
    dataset = pd.read_csv(args.dataset, index_col=0, keep_default_na=False)

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

    y = utils.get_onehot_from_labels(dataset["PREDICATE_LABEL"])

    assert dataset.shape[0] == y.shape[0]

    labels2preds = {i: lab for (i, lab) in enumerate(unique_classes)}
    print(labels2preds)
    docs = dataset["SENTENCE"].values
    scores = bert.predict(docs, predict_classes=False).numpy()
    utils.save_predictions(dataset, scores, args.outfile, labels2preds)


if __name__ == "__main__":
    args = parse_args()
    main(args)
