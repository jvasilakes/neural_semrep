import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter

import bert_models
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True,
                        help="""Directory containing dataset splits as
                                {train,val,test}.csv""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="""Where to save the model predictions.""")
    parser.add_argument("--bert_model_class", type=str, required=True,
                        choices=["pooled", "entity"],
                        help="""BERTModel subclass from bert_models.py
                                to use.""")
    parser.add_argument("--bert_model_file", type=str, required=True,
                        help="""tfhub.dev URL for file path to
                                saved BERT model compatible with
                                tensorflow_hub.KerasLayer""")
    parser.add_argument("--no_finetune", action="store_true", default=False,
                        help="""If specified, don't fine-tune
                                the BERT layer.""")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="""Set the global random seed.""")
    parser.add_argument("--checkpoint_dirname", type=str, default=None,
                        help="""Name of directory within outdir in which to
                                save the model checkpoints.""")
    parser.add_argument("--tensorboard_dirname", type=str, default=None,
                        help="""Name of directory within outdir in which to
                                save the tensorboard logs.""")
    parser.add_argument("--epochs", type=int, default=1,
                        help="""Number of epochs to train for.""")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2],
                        default=1, help="Passed to tf.Model.fit()")
    return parser.parse_args()


def main(args):
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    # Read the datafiles.
    train_file = os.path.join(args.datadir, "train.csv")
    val_file = os.path.join(args.datadir, "val.csv")
    test_file = os.path.join(args.datadir, "test.csv")
    train_data = pd.read_csv(train_file, index_col=0, keep_default_na=False)
    val_data = pd.read_csv(val_file, index_col=0, keep_default_na=False)
    test_data = pd.read_csv(test_file, index_col=0, keep_default_na=False)

    # BERT requires the [SEP] token at the end, so we add it here.
    train_data.loc[:, "SENTENCE"] = train_data["SENTENCE"].apply(
            lambda s: s + " [SEP]")
    val_data.loc[:, "SENTENCE"] = val_data["SENTENCE"].apply(
            lambda s: s + " [SEP]")
    test_data.loc[:, "SENTENCE"] = test_data["SENTENCE"].apply(
            lambda s: s + " [SEP]")

    # Conver the integer labels to one-hot vectors.
    train_docs = train_data["SENTENCE"].values
    train_y = utils.get_onehot_from_labels(train_data["PREDICATE_LABEL"])
    val_docs = val_data["SENTENCE"].values
    val_y = utils.get_onehot_from_labels(val_data["PREDICATE_LABEL"])
    test_docs = test_data["SENTENCE"].values

    # Compute the initial biases for the output nodes
    # according to the distribution of labels in the training data.
    train_counts = Counter(train_data["PREDICATE"])
    initial_biases = utils.compute_initial_bias(train_y)
    print("Initial Biases")
    print("--------------")
    for ((pred, count), bias) in zip(train_counts.items(), initial_biases):
        print(f"{pred:<15} {count:<5}: {bias:.4f}")

    # Get the specific BERT model to use.
    bert_class = get_bert_model_class(args.bert_model_class)
    print()
    print(bert_class)

    # Create the log directories for BERT.
    os.makedirs(args.outdir, exist_ok=False)
    ckpt_dir = tb_logdir = None
    if args.checkpoint_dirname is not None:
        ckpt_dir = os.path.join(args.outdir, args.checkpoint_dirname)
        os.makedirs(ckpt_dir, exist_ok=False)
    if args.tensorboard_dirname is not None:
        tb_logdir = os.path.join(args.outdir, args.tensorboard_dirname)
        os.makedirs(tb_logdir, exist_ok=False)

    # Instantiate the BERT model class.
    n_classes = train_data["PREDICATE_LABEL"].unique().shape[0]
    bert_trainable = not args.no_finetune
    bert = bert_class(n_classes=n_classes,
                      bert_model_file=args.bert_model_file,
                      bert_trainable=bert_trainable,
                      max_seq_length=256, batch_size=16,
                      dropout_rate=0.2, learning_rate=2e-5,
                      random_seed=args.random_seed,
                      initial_biases=initial_biases,
                      validation_data=(val_docs, val_y),
                      fit_metrics=["precision", "recall"],
                      checkpoint_dir=ckpt_dir,
                      tensorboard_logdir=tb_logdir)
    config_outf = os.path.join(args.outdir, f"{bert.name}.json")
    bert.save_config(config_outf)

    # Given the initial biases computed above, check that the model's
    # initial loss is close to the expected.
    initial_loss = bert.compute_loss(train_docs, train_y)[0]
    expected_loss = utils.compute_expected_loss_categorical(train_y)
    print("--------------------------------")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Expected loss: {expected_loss:.4f}")
    print("--------------------------------")

    # Train.
    bert.fit(train_docs, train_y, epochs=args.epochs, verbose=args.verbose)

    # Predict on the train, val, and test data.
    # To do this, we need a mapping from integer labels (the BERT output)
    # to predicates.
    labels2preds = train_data[["PREDICATE_LABEL", "PREDICATE"]].drop_duplicates()  # noqa
    labels2preds = {row.PREDICATE_LABEL: row.PREDICATE
                    for row in labels2preds.itertuples()}

    train_preds_outfile = os.path.join(args.outdir, "predictions_train.csv")
    train_scores = bert.predict(train_docs, predict_classes=False).numpy()
    utils.save_predictions(train_data, train_scores,
                           train_preds_outfile, labels2preds)

    val_preds_outfile = os.path.join(args.outdir, "predictions_val.csv")
    val_scores = bert.predict(val_docs, predict_classes=False).numpy()
    utils.save_predictions(val_data, val_scores,
                           val_preds_outfile, labels2preds)

    test_preds_outfile = os.path.join(args.outdir, "predictions_test.csv")
    test_scores = bert.predict(test_docs, predict_classes=False).numpy()
    utils.save_predictions(test_data, test_scores,
                           test_preds_outfile, labels2preds)


def get_bert_model_class(bert_model_string):
    lookup = {
              "pooled": bert_models.PooledModel,
              "entity": bert_models.EntityModel,
             }
    return lookup[bert_model_string.lower()]


if __name__ == "__main__":
    args = parse_args()
    main(args)
