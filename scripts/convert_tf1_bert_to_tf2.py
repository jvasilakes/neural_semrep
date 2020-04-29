import argparse
import pandas as pd
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf2_bert_file", type=str, required=True)
    parser.add_argument("--tf1_ckpt_file", type=str, required=True)
    parser.add_argument("--tf2_to_tf1_name_mapping", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


def main(tf2_bert_file, tf1_ckpt_file, tf2_to_tf1_name_mapping, outdir):
    tf2_module = tf.saved_model.load(tf2_bert_file)
    tf1_ckpt = tf.train.load_checkpoint(tf1_ckpt_file)

    tf2_to_tf1_df = pd.read_csv(tf2_to_tf1_name_mapping)
    tf2_to_tf1_varnames = dict(zip(tf2_to_tf1_df["TF2"].values,
                                   tf2_to_tf1_df["TF1"].values))

    assignment_map = {}
    for (i, tf2_var) in enumerate(tf2_module.trainable_variables):
        tf2_name = tf2_var.name.split(':')[0]
        tf1_name = tf2_to_tf1_varnames[tf2_name]
        ckpt_value = tf1_ckpt.get_tensor(tf1_name)
        if tf2_var.shape != ckpt_value.shape:
            print("Replacing first with second.")
            print(tf2_var.shape)
            print(ckpt_value.shape)
            new_tf2_var = tf.Variable(ckpt_value,
                                      trainable=tf2_var.trainable,
                                      dtype=tf2_var.dtype)
            tf2_module.trainable_variables[i] = new_tf2_var
            print(tf2_module.trainable_variables[i].shape)
            input()
        assignment_map[tf1_name] = tf2_var

    tf.compat.v1.train.init_from_checkpoint(tf1_ckpt_file, assignment_map)
    tf.saved_model.save(tf2_module, outdir)


if __name__ == "__main__":
    args = parse_args()
    main(args.tf2_bert_file, args.tf1_ckpt_file,
         args.tf2_to_tf1_name_mapping, args.outdir)
