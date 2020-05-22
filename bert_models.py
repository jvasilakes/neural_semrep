import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from collections import Counter

import bert_tokenization


class LearningRateLogCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir):
        super(LearningRateLogCallback, self).__init__()
        file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        file_writer.set_as_default()
        self.batch = 0

    def on_train_batch_end(self, _, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError("Optimizer must have an 'learning_rate' attribute.")  # noqa
        logs = logs or {}
        learning_rate_fn = tf.keras.backend.get_value(
            self.model.optimizer.learning_rate)
        learning_rate = learning_rate_fn(self.batch)
        tf.summary.scalar("learning rate", data=learning_rate)
        self.batch += 1


class MaskReduceLayer(tf.keras.layers.Layer):
    """
    Optionally applies a boolean mask to the time
    dimension of the NxK input tensor, then
    sums the components to return a K dimensional vector.

    input = [I, ate, dinner]
    mask = [True, False, True]
    onehot_input = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
    masked = MaskReduceLayer()(onehot_input, mask)
    print(mask)

    [1, 0, 1]  # Elementwise sum of 'I' and 'dinner'
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            masked = inputs
        else:
            masked = tf.ragged.boolean_mask(inputs, mask)
        # return tf.reduce_sum(masked, axis=1)
        # return sqrtn(masked, axis=1)  # TODO: Implement this.
        return tf.reduce_mean(masked, axis=1)


class BERTModel(object):
    """
    The generic BERT-based model. Subclass to implement
    your chosen architecture.
    :param int n_classes: Number of output classes. Use 1 or 2 for binary.
    :param str bert_model_file: Path to tensorflow_hub compatible BERT model.
    :param int max_seq_length: (Default 128) Maximum input sequence length.
    :param int batch_size: (Default 16) Batch size.
    :param float dropout_rate: (Default 0.2) Dropout rate for dropout layer
                               after BERT layer.
    :param float learning_rate: (Default 2e-5) The learning rate.
    :param int random_seed: Set the tensorflow random seed.
    :param bool bert_trainable: (Default True) Whether to allow fine-tuning
                                of the BERT layer.
    :param tuple(np.array) validation_data: (X, y) tuple of validation data.
    :param list(str) fit_metrics: (Default ['accuracy']) List of metrics to
                                 compute during training.
    :param np.array(float) initial_biases: (Default None) If None, use 'zeros'.
                                         Otherwise, initialize with the given
                                         constant values.
    :param str checkpoint_dir: (Default None) Path to directory in which to
                               save model checkpoints. Creates it if it
                               doesn't exist. If None, don't save checkpoints.
    :param str tensorboard_logdir: (Default None) Path to directory in which to
                               save tensorboard logs. Creates it if it
                               doesn't exist. If None, don't save logs.
    """

    name = "BERTModel"

    def __init__(self, n_classes, bert_model_file, max_seq_length=128,
                 batch_size=16, dropout_rate=0.2, learning_rate=2e-5,
                 random_seed=None, bert_trainable=True, validation_data=None,
                 fit_metrics=["accuracy"], initial_biases=None,
                 checkpoint_dir=None, tensorboard_logdir=None):
        if random_seed is not None:
            self.random_seed = random_seed
            tf.random.set_seed(random_seed)
        self.n_classes = n_classes
        self.bert_model_file = bert_model_file
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = self._get_learning_rate(learning_rate)
        self.bert_trainable = bert_trainable
        self.fit_metrics = self._get_metrics(fit_metrics)
        self.initial_biases = initial_biases
        self.bias_initializer = self._get_bias_initializer()
        self._callbacks = []
        if checkpoint_dir is not None:
            print("Writing model checkpoints to: ", checkpoint_dir)
            ckpnt_cb = self._get_checkpoint_callback(checkpoint_dir)
            self._callbacks.append(ckpnt_cb)
        if tensorboard_logdir is not None:
            print("Writing Tensorboard logs to: ", tensorboard_logdir)
            tb_cb = self._get_tensorboard_callback(tensorboard_logdir)
            self._callbacks.append(tb_cb)
            if isinstance(self.learning_rate,
                          tf.keras.optimizers.schedules.LearningRateSchedule):
                lr_log_callback = self._get_learning_rate_log_callback(
                        tensorboard_logdir)
                self._callbacks.append(lr_log_callback)
        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()
        self.validation_data = validation_data
        if self.validation_data is not None:
            self.validation_data = self._prep_validation_data(validation_data)
        self.config = self._collect_config()
        for (key, value) in self.config.items():
            print(f"{key}: {value}")

    def get_model(self, *args, **kwargs):
        """
        Implement subclass with your desired architecture.
        """
        raise NotImplementedError

    def tokenize(self, *args, **kwargs):
        """
        Implement subclass with tokenization for your desired architecture.
        """
        raise NotImplementedError

    @classmethod
    def from_model_checkpoint(cls, config_file, weights_file):
        config = json.load(open(config_file))
        bert_model = cls(**config)
        bert_model.model.load_weights(weights_file)
        return bert_model

    def _get_learning_rate(self, learning_rate):
        if isinstance(learning_rate, str):
            if learning_rate.lower() == "exponential_decay":
                scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=2e-5, decay_rate=0.96,
                        decay_steps=300)
            elif learning_rate.lower() == "polynomial_decay":
                scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
                        initial_learning_rate=2e-5, decay_rate=0.96,
                        end_learning_rate=0.0, power=2)
            else:
                raise ValueError(f"Unknown schedule '{learning_rate}'")
            return scheduler
        else:
            return float(learning_rate)

    def _get_bias_initializer(self):
        if self.initial_biases is None:
            return "zeros"
        return tf.keras.initializers.Constant(self.initial_biases)

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

    def _collect_config(self):
        config = dict(n_classes=self.n_classes,
                      bert_model_file=self.bert_model_file,
                      max_seq_length=self.max_seq_length,
                      batch_size=self.batch_size,
                      learning_rate=self.learning_rate,
                      bert_trainable=self.bert_trainable,
                      random_seed=self.random_seed,
                      dropout_rate=self.dropout_rate,
                      fit_metrics=[m.name for m in self.fit_metrics],
                      initial_biases=list(self.initial_biases))
        return config

    def save_config(self, outfile):
        if self.config is None:
            raise ValueError("Config not defined yet. Please run __init__")
        with open(outfile, 'w') as outF:
            json.dump(self.config, outF)

    def _get_checkpoint_callback(self, checkpoint_dir):
        fname = "weights.{epoch:02d}-{loss:.2f}.hdf5"
        filepath = os.path.join(checkpoint_dir, fname)
        return tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                  monitor="val_loss",
                                                  save_weights_only=True,
                                                  verbose=1)

    def _get_tensorboard_callback(self, tensorboard_logdir):
        tb_cb = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_logdir, update_freq="batch",
            write_graph=True, write_images=True)
        return tb_cb

    def _get_learning_rate_log_callback(self, tensorboard_logdir):
        lr_cb = LearningRateLogCallback(tensorboard_logdir)
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

    def get_token_masks(self, tokens):
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

    def compute_loss(self, documents, labels):
        processed_documents = self.tokenize(documents)
        loss = self.model.evaluate(processed_documents, labels,
                                   batch_size=self.batch_size,
                                   verbose=0)
        return loss

    def fit(self, documents, labels, epochs=1, verbose=1, **tokenize_kwargs):
        processed_documents = self.tokenize(documents, **tokenize_kwargs)
        self.train_history = self.model.fit(
                processed_documents, labels, batch_size=self.batch_size,
                epochs=epochs, validation_data=self.validation_data,
                callbacks=self._callbacks, verbose=verbose)

    def predict(self, documents, predict_classes=True, **tokenize_kwargs):
        processed_documents = self.tokenize(documents, **tokenize_kwargs)
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


class PooledModel(BERTModel):
    """
    BERT pooled output -> dropout -> dense

    :param int n_classes: Number of output classes. Use 1 or 2 for binary.
    :param str bert_model_file: Path to tensorflow_hub compatible BERT model.
    :param int max_seq_length: (Default 128) Maximum input sequence length.
    :param int batch_size: (Default 16) Batch size.
    :param float dropout_rate: (Default 0.2) Dropout rate for dropout layer
                               after BERT layer.
    :param float learning_rate: (Default 2e-5) The learning rate.
    :param int random_seed: Set the tensorflow random seed.
    :param bool bert_trainable: (Default True) Whether to allow fine-tuning
                                of the BERT layer.
    :param tuple(np.array) validation_data: (X, y) tuple of validation data.
    :param list(str) fit_metrics: (Default ['accuracy']) List of metrics to
                                 compute during training.
    :param np.array(float) initial_biases: (Default None) If None, use 'zeros'.
                                         Otherwise, initialize with the given
                                         constant values.
    :param str checkpoint_dir: (Default None) Path to directory in which to
                               save model checkpoints. Creates it if it
                               doesn't exist. If None, don't save checkpoints.
    :param str tensorboard_logdir: (Default None) Path to directory in which to
                               save tensorboard logs. Creates it if it
                               doesn't exist. If None, don't save logs.
    """

    name = "PooledModel"

    def __init__(self, *args, **kwargs):
        print(f"Initializing {self.name}")
        super().__init__(*args, **kwargs)

    def tokenize(self, documents):
        self._all_raw_tokens = []
        all_token_ids = []
        all_token_masks = []
        all_token_segments = []
        for doc in documents:
            first, _, scnd = doc.partition("[SEP]")
            first_toks = self.tokenizer.tokenize(first)
            scnd_toks = self.tokenizer.tokenize(scnd)

            # subtract 2 for [CLS], [SEP]
            first_maxlen = self.max_seq_length - len(scnd_toks) - 2
            first_toks = self.cut_tokens(first_toks, first_maxlen)
            tokens = ["[CLS]"] + first_toks + ["[SEP]"] + scnd_toks
            self._all_raw_tokens.append(tokens)
            # Format as three arrays as expected by BERT
            token_ids = self.get_token_ids(tokens)
            token_masks = self.get_token_masks(tokens)
            segments = self.get_segments(tokens)
            all_token_ids.append(token_ids)
            all_token_masks.append(token_masks)
            all_token_segments.append(segments)
        return_val = [np.array(all_token_ids),
                      np.array(all_token_masks),
                      np.array(all_token_segments)]
        return return_val

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

        bert_layer = hub.KerasLayer(self.bert_model_file,
                                    trainable=self.bert_trainable,
                                    name="bert")

        pooled_output, seq_output = bert_layer(bert_inputs)
        # dense_input = seq_output[:, 0, :]  # [CLS] output
        dense_input = pooled_output
        drop_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)(dense_input)  # noqa
        pred_layer = tf.keras.layers.Dense(
                self.n_classes, activation=out_activation,
                bias_initializer=self.bias_initializer,
                name="prediction")(drop_layer)

        model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred_layer)
        optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate, epsilon=1e-8)
        model.compile(loss=loss, optimizer=optimizer, metrics=self.fit_metrics)
        return model


class EntityModel(BERTModel):
    """
    BERT sequence_output -> subject and object context embeddings ->
    dropout -> dense
    I.e.

    '[SUBJ] went to the [OBJ] yesterday'    # Input sentence
                 |
                BERT                        # BERT layer
                 |
    [[float float float float float float]  # Context embeddings
                  ...                       # (batch_size, max_seq_lenth, 768)
     [float float float float float float]]
                 |
    [  1     0     0     0     0     0  ]   # [SUBJ] mask (masks subject in context emb)  # noqa
    [  0     0     0     1     0     0  ]   # [OBJ] mask  (masks object in context emb)   # noqa
                 |
          [float_subj, float_obj]        # Element-wise sum over subj/obj tokens followed by concatenate  # noqa
                 |
             Dropout + Dense

    :param int n_classes: Number of output classes. Use 1 or 2 for binary.
    :param str bert_model_file: Path to tensorflow_hub compatible BERT model.
    :param int max_seq_length: (Default 128) Maximum input sequence length.
    :param int batch_size: (Default 16) Batch size.
    :param float dropout_rate: (Default 0.2) Dropout rate for dropout layer
                               after BERT layer.
    :param float learning_rate: (Default 2e-5) The learning rate.
    :param int random_seed: Set the tensorflow random seed.
    :param bool bert_trainable: (Default True) Whether to allow fine-tuning
                                of the BERT layer.
    :param tuple(np.array) validation_data: (X, y) tuple of validation data.
    :param list(str) fit_metrics: (Default ['accuracy']) List of metrics to
                                 compute during training.
    :param np.array(float) initial_biases: (Default None) If None, use 'zeros'.
                                         Otherwise, initialize with the given
                                         constant values.
    :param str checkpoint_dir: (Default None) Path to directory in which to
                               save model checkpoints. Creates it if it
                               doesn't exist. If None, don't save checkpoints.
    :param str tensorboard_logdir: (Default None) Path to directory in which to
                               save tensorboard logs. Creates it if it
                               doesn't exist. If None, don't save logs.
    :param list(str) mask_tokens: (Default ['[ARG1]', '[ARG2]']) DANGER!
                                  Changing this from the default can result
                                  in unexpected bad behavior or errors!
                                  A list of length 2 defining the tokens
                                  used to mask the subject and objects of
                                  a predication in the given sentence.
    """

    name = "EntityModel"

    def __init__(self, mask_tokens=["[ARG1]", "[ARG2]"], *args, **kwargs):
        self.mask_tokens = mask_tokens
        self.total_mask_length = 10
        print(f"Initializing {self.name}")
        super().__init__(*args, **kwargs)

    def _get_subj_obj_masks(self, tokens):
        """Returns indices in ragne(max_seq_length)
           of [SUBJ] and [OBJ] tokens. Used by
           EntityModel.tokenize()."""
        if len(tokens) > self.max_seq_length:
            raise IndexError("Token length more than max seq length!")
        subj_toks = self.tokenizer.tokenize(self.mask_tokens[0])
        obj_toks = self.tokenizer.tokenize(self.mask_tokens[1])
        assert len(subj_toks) + len(obj_toks) == self.total_mask_length

        subj_mask = np.zeros(shape=(self.max_seq_length,), dtype=int)
        obj_mask = np.zeros(shape=(self.max_seq_length,), dtype=int)

        def getsubidx(x, y):
            lx, ly = len(x), len(y)
            for i in range(lx):
                if x[i:i+ly] == y:
                    return i

        subj_start = getsubidx(tokens, subj_toks)
        if subj_start is not None:
            subj_end = subj_start + len(subj_toks)
            subj_mask[subj_start:subj_end] = 1
        obj_start = getsubidx(tokens, obj_toks)
        if obj_start is not None:
            obj_end = obj_start + len(obj_toks)
            obj_mask[obj_start:obj_end] = 1
        return (subj_mask, obj_mask)

    def tokenize(self, documents):
        """Adds mask inputs."""
        self._all_raw_tokens = []
        all_token_ids = []
        all_token_masks = []
        all_token_segments = []
        all_subj_masks = []
        all_obj_masks = []
        for doc in documents:
            first, _, scnd = doc.partition("[SEP]")
            first_toks = self.tokenizer.tokenize(first)
            scnd_toks = self.tokenizer.tokenize(scnd)

            # subtract 2 for [CLS], [SEP]
            first_maxlen = self.max_seq_length - len(scnd_toks) - 2
            first_toks = self.cut_tokens(first_toks, first_maxlen)
            tokens = ["[CLS]"] + first_toks + ["[SEP]"] + scnd_toks
            self._all_raw_tokens.append(tokens)
            # Format as three arrays as expected by BERT
            token_ids = self.get_token_ids(tokens)
            token_masks = self.get_token_masks(tokens)
            segments = self.get_segments(tokens)
            subj_mask, obj_mask = self._get_subj_obj_masks(tokens)
            all_token_ids.append(token_ids)
            all_token_masks.append(token_masks)
            all_token_segments.append(segments)
            all_subj_masks.append(subj_mask)
            all_obj_masks.append(obj_mask)
        return_val = [np.array(all_token_ids),
                      np.array(all_token_masks),
                      np.array(all_token_segments),
                      np.array(all_subj_masks),
                      np.array(all_obj_masks)]
        return return_val

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

        subj_mask = tf.keras.layers.Input(shape=inshape,
                                          dtype=tf.bool,
                                          name="subject_mask")
        obj_mask = tf.keras.layers.Input(shape=inshape,
                                         dtype=tf.bool,
                                         name="object_mask")
        mask_inputs = [subj_mask, obj_mask]

        bert_layer = hub.KerasLayer(self.bert_model_file,
                                    trainable=self.bert_trainable,
                                    name="bert")

        pooled_output, seq_output = bert_layer(bert_inputs)
        # We mask the subject and object individually to control their
        # order when passed to the Dense layer.
        subj = MaskReduceLayer(name="masked_subject")(seq_output, subj_mask)
        obj = MaskReduceLayer(name="masked_object")(seq_output, obj_mask)
        dense_input = tf.keras.layers.Concatenate()([subj, obj])
        drop_layer = tf.keras.layers.Dropout(
                rate=self.dropout_rate, name="dropout")(dense_input)
        pred_layer = tf.keras.layers.Dense(
                self.n_classes, activation=out_activation,
                bias_initializer=self.bias_initializer,
                name="prediction")(drop_layer)

        model_inputs = bert_inputs + mask_inputs
        model = tf.keras.models.Model(inputs=model_inputs, outputs=pred_layer)
        optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate, epsilon=1e-8)
        model.compile(loss=loss, optimizer=optimizer, metrics=self.fit_metrics)
        return model

    def _collect_config(self):
        config = dict(n_classes=self.n_classes,
                      bert_model_file=self.bert_model_file,
                      max_seq_length=self.max_seq_length,
                      batch_size=self.batch_size,
                      learning_rate=self.learning_rate,
                      bert_trainable=self.bert_trainable,
                      random_seed=self.random_seed,
                      dropout_rate=self.dropout_rate,
                      fit_metrics=[m.name for m in self.fit_metrics],
                      initial_biases=list(self.initial_biases),
                      mask_tokens=self.mask_tokens)
        return config
