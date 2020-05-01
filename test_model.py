import argparse
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras.backend import get_graph


parser = argparse.ArgumentParser()
parser.add_argument("outdir", type=str)
args = parser.parse_args()


inshape = (128,)
input_word_ids = tf.keras.layers.Input(shape=inshape,
                                       dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=inshape,
                                   dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=inshape,
                                    dtype=tf.int32,
                                    name="segment_ids")
bert_inputs = [input_word_ids, input_mask, segment_ids]

subj_mask = tf.keras.layers.Input(shape=inshape,
                                  dtype=tf.bool,
                                  name="subj_mask")
obj_mask = tf.keras.layers.Input(shape=inshape,
                                 dtype=tf.bool,
                                 name="obj_mask")
index_inputs = [subj_mask, obj_mask]

bert_layer = hub.KerasLayer("models/bert_en_cased_L-12_H-768_A-12",
                            trainable=True, name="bert")

# pooled_output: [batch_size, 768]
# seq_output: [batch_size, max_seq_length, 768]
pooled_output, seq_output = bert_layer(bert_inputs)
subj = tf.boolean_mask(seq_output, subj_mask, axis=0, name="subj_mask")
obj = tf.boolean_mask(seq_output, obj_mask, axis=0, name="obj_mask")
dense_input = tf.stack([subj, obj], axis=1)
#seq = tf.keras.layers.Concatenate(axis=1)([seq1, seq2])
pred_layer = tf.keras.layers.Dense(
        4, activation="softmax",
        name="prediction")(dense_input)

model_inputs = bert_inputs + index_inputs
#model_inputs = bert_inputs
model = tf.keras.models.Model(inputs=model_inputs, outputs=pred_layer)
model.compile(loss="categorical_crossentropy", optimizer="adam")

print(model.summary())

tb_writer = tf.summary.create_file_writer(args.outdir)
with tb_writer.as_default():
    if not model.run_eagerly:
        summary_ops_v2.graph(get_graph(), step=0)
        print("Done")
