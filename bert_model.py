import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from collections import Counter

import bert_tokenization

tf.random.set_seed(42)


class LearningRateLogCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir):
        super(LearningRateLogCallback, self).__init__()
        file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        file_writer.set_as_default()

    def on_train_batch_end(self, batch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError("Optimizer must have an 'learning_rate' attribute.")  # noqa
        logs = logs or {}
        learning_rate_fn = tf.keras.backend.get_value(
            self.model.optimizer.learning_rate)
        learning_rate = learning_rate_fn(batch)
        tf.summary.scalar("learning rate", data=learning_rate)


class BERTModel(object):

    def __init__(self, n_classes, bert_file, max_seq_length=128, batch_size=16,
                 dropout_rate=0.2, learning_rate=2e-5,
                 validation_data=None, fit_metrics=["accuracy"],
                 initial_bias=None, checkpoint_dir=None, logdir=None):
        self.n_classes = n_classes
        self.bert_file = bert_file
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.fit_metrics = self._get_metrics(fit_metrics)
        self.initial_bias = initial_bias
        self.bias_initializer = self._get_bias_initializer()
        self._callbacks = []
        if checkpoint_dir is not None:
            print(checkpoint_dir)
            self._save_params(checkpoint_dir)
            ckpnt_cb = self._get_checkpoint_callback(checkpoint_dir)
            self._callbacks.append(ckpnt_cb)
        if logdir is not None:
            print(logdir)
            tb_cb = self._get_tensorboard_callback(logdir)
            self._callbacks.append(tb_cb)
            lr_log_callback = self._get_learning_rate_log_callback(logdir)
            self._callbacks.append(lr_log_callback)
        print(self.bert_file)
        print(self.max_seq_length)
        print(self.batch_size)
        print("getting model...", end='')
        self.model = self.get_model()
        print("done")
        print("getting tokenizer...", end='')
        self.tokenizer = self.get_tokenizer()
        print("done")
        self.validation_data = validation_data
        if self.validation_data is not None:
            self.validation_data = self._prep_validation_data(validation_data)

    @classmethod
    def from_model_checkpoint(cls, params_file, weights_file):
        params = json.load(open(params_file))
        bert_model = cls(**params)
        bert_model.model.load_weights(weights_file)
        return bert_model

    def _get_bias_initializer(self):
        if self.initial_bias is None:
            return "zeros"
        return tf.keras.initializers.Constant(self.initial_bias)

    def _get_metrics(self, metrics):
        metric_fns = []
        for m in metrics:
            if m == "accuracy":
                metric_fns.append(tf.keras.metrics.Accuracy())
            elif m == "precision":
                metric_fns.append(tf.keras.metrics.Precision())
            elif m == "recall":
                metric_fns.append(tf.keras.metrics.Recall())
            else:
                raise ValueError(f"Unsupported metric '{m}'")
        return metric_fns

    def _save_params(self, outdir):
        outfile = os.path.join(outdir, "model.json")
        params = dict(bert_file=self.bert_file,
                      max_seq_length=self.max_seq_length,
                      batch_size=self.batch_size,
                      fit_metrics=[m.name for m in self.fit_metrics],
                      initial_bias=float(self.initial_bias))
        with open(outfile, 'w') as outF:
            json.dump(params, outF)

    def _get_checkpoint_callback(self, checkpoint_dir):
        fname = "weights.{epoch:02d}-{loss:.2f}.hdf5"
        filepath = os.path.join(checkpoint_dir, fname)
        return tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                  monitor="loss",
                                                  save_weights_only=True,
                                                  verbose=1)

    def _get_tensorboard_callback(self, logdir):
        tb_cb = tf.keras.callbacks.TensorBoard(
            log_dir=logdir, update_freq="batch",
            write_graph=True, write_images=True)
        return tb_cb

    def _get_learning_rate_log_callback(self, logdir):
        lr_cb = LearningRateLogCallback(logdir)
        return lr_cb

    def _prep_validation_data(self, validation_data):
        text = validation_data[0]
        labels = validation_data[1]
        processed_text = self.tokenize(text)
        return processed_text, labels

    def cut_tokens(self, tokens, length):
        tokens = tokens[:length]
        return tokens

    def get_token_ids(self, tokens):
        """Token ids from Tokenizer vocab"""
        if len(tokens) > self.max_seq_length:
            raise IndexError("Token length more than max seq length!")
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (self.max_seq_length - len(token_ids))
        return np.array(input_ids)

    def get_masks(self, tokens):
        """Mask for padding"""
        if len(tokens) > self.max_seq_length:
            raise IndexError("Token length more than max seq length!")
        masks = [1] * len(tokens) + [0] * (self.max_seq_length - len(tokens))
        return np.array(masks)

    def get_segments(self, tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens) > self.max_seq_length:
            raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        all_segments = segments + [0] * (self.max_seq_length - len(tokens))
        return np.array(all_segments)

    def get_model(self):
        if self.n_classes in [1, 2]:
            out_activation = "sigmoid"
            loss = "binary_crossentropy"
        else:
            out_activation = "softmax"
            loss = "categorical_crossentropy"

        inshape = (self.max_seq_length, )
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

        bert_layer = hub.KerasLayer(self.bert_file,
                                    trainable=True, name="bert")

        pooled_output, seq_output = bert_layer(bert_inputs)
        # dense_input = seq_output[:, 0, :]  # [CLS] output
        dense_input = pooled_output
        drop_layer = tf.keras.layers.Dropout(rate=0.2)(dense_input)
        pred_layer = tf.keras.layers.Dense(
                self.n_classes, activation=out_activation,
                bias_initializer=self.bias_initializer,
                name="prediction")(drop_layer)

        model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred_layer)
        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_rate=0.96, decay_steps=300)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8)  # noqa
        # optimizer = tf.keras.optimizers.Adam(
        #         learning_rate=scheduler, epsilon=1e-8)
        optimizer = tf.keras.optimizers.SGD(learning_rate=scheduler)
        model.compile(loss=loss, optimizer=optimizer, metrics=self.fit_metrics)
        return model

    def get_tokenizer(self):
        bert_layer = self.model.get_layer("bert")
        try:
            vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()  # noqa
            do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
            tokenizer = bert_tokenization.FullTokenizer(vocab_file,
                                                        do_lower_case)
        # AlBERT uses a different tokenizer.
        # AttributeError: '_UserObject' object has no attribute 'vocab_file'
        except AttributeError:
            sp_model_file = bert_layer.resolved_object.sp_model_file.asset_path.numpy()  # noqa
            tokenizer = bert_tokenization.FullSentencePieceTokenizer(
                    sp_model_file)
        return tokenizer

    def tokenize(self, documents):
        all_token_ids = []
        all_token_masks = []
        all_token_segments = []
        for doc in documents:
            first, _, scnd = doc.partition("[SEP]")
            scnd_toks = self.tokenizer.tokenize(scnd)
            first_toks = self.tokenizer.tokenize(first)
            # subtract 2 for [CLS], [SEP]
            first_maxlen = self.max_seq_length - len(scnd_toks) - 2
            first_toks = self.cut_tokens(first_toks, first_maxlen)
            tokens = ["[CLS]"] + first_toks + ["[SEP]"] + scnd_toks
            # Format as three arrays as expected by BERT
            token_ids = self.get_token_ids(tokens)
            token_masks = self.get_masks(tokens)
            segments = self.get_segments(tokens)
            all_token_ids.append(token_ids)
            all_token_masks.append(token_masks)
            all_token_segments.append(segments)
        return [np.array(all_token_ids),
                np.array(all_token_masks),
                np.array(all_token_segments)]

    def compute_loss(self, documents, labels):
        processed_documents = self.tokenize(documents)
        loss = self.model.evaluate(processed_documents, labels,
                                   batch_size=self.batch_size,
                                   verbose=0)
        return loss

    def fit(self, documents, labels, epochs=1):
        processed_documents = self.tokenize(documents)
        self.train_history = self.model.fit(
                processed_documents, labels, batch_size=self.batch_size,
                epochs=epochs, validation_data=self.validation_data,
                callbacks=self._callbacks)

    def predict(self, documents, predict_classes=True):
        processed_documents = self.tokenize(documents)
        scores = self.model.predict(processed_documents,
                                    batch_size=self.batch_size)
        scores = tf.squeeze(scores)
        if predict_classes is True:
            predictions = tf.cast(scores >= 0.5, dtype=tf.int32)
            return predictions
        return scores

    def _compute_sample_weights(self, y):

        def weight(count, total):
            return (1 / count) * (total / 2)

        total = y.shape[0]
        counts = Counter(y)
        weights = {l: weight(c, total) for (l, c) in counts.items()}
        print("Sample weights: ", weights)
        weights = np.array([weights[l] for l in y])
        return weights