from data import Task
import data
import numpy as np
import tensorflow as tf
from pathlib2 import Path
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.framework import ops


class Model(object):
    def __init__(self, embedding, task, params, session=None, graph=None, max_to_keep=1000):
        # just to specify which model; not related to tf.train.get_or_create_global_step()
        self.global_step = 0
        self.graph = graph or tf.get_default_graph()
        self.session = session or tf.Session()
        self.task = task
        self.turns = tf.placeholder(
            shape=(None, None, None), dtype=tf.int32, name="turns")
        self.senders = tf.placeholder(
            shape=(None, None), dtype=tf.bool, name="senders")
        self.turn_lengths = tf.placeholder(
            shape=(None, None), dtype=tf.int32, name="turn_lengths")
        self.dialogue_lengths = tf.placeholder(
            shape=(None), dtype=tf.int32, name="dialogue_lengths")
        self.h_nuggets_labels = tf.placeholder(
            shape=(None, None, len(data.HELPDESK_NUGGET_TYPES_WITH_PAD)),
            dtype=tf.float32, name="helpdesk_nuggets_labels")
        self.c_nuggets_labels = tf.placeholder(
            shape=(None, None, len(data.CUSTOMER_NUGGET_TYPES_WITH_PAD)),
            dtype=tf.float32, name="customer_nuggets_labels")
        self.quality_labels = tf.placeholder(
            shape=(None, len(data.QUALITY_MEASURES), len(data.QUALITY_SCALES)),
            dtype=tf.float32, name="quality_labels")
        self.dropout = tf.placeholder_with_default(
            params.dropout, shape=[], name="dropout_rate")

        self._set_operations(embedding, task, params)

        # embedding weights are not saved into Graph if it is not trainable
        self.saver = tf.train.Saver(
            tf.trainable_variables(), save_relative_paths=True, max_to_keep=max_to_keep)

        self.run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE) if params.trace else tf.RunOptions()
        self.run_metadata = tf.RunMetadata()

        self.session.run(tf.global_variables_initializer())
        # Variable init does not accept weights that are larger than 2GB
        # This must be ran after global_variables_initializer
        self.session.run(self.embedding.initializer, feed_dict={
                         self.embedding.initial_value: embedding})

        assert np.allclose(self.session.run(self.embedding), embedding)

    def _set_operations(self, embedding, task, params):
        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(shape=embedding.shape, trainable=params.update_embedding,
                                             name="embedding_weights", dtype=tf.float32)
            turns_embedded = tf.nn.embedding_lookup(self.embedding, self.turns)
        turns_boW = tf.reduce_sum(
            turns_embedded, axis=2, name="BoW")  # Bag of Words

        features = (turns_boW, self.senders,
                    self.turn_lengths, self.dialogue_lengths)

        if task == Task.nugget:
            self.c_nuggets_logits, self.h_nuggets_logits = nugget_model_fn(
                features, self.dropout, params)
            self.loss = nugget_loss(
                self.c_nuggets_logits, self.h_nuggets_logits,
                self.c_nuggets_labels, self.h_nuggets_labels, self.dialogue_lengths, tf.shape(self.turns)[1])

            self.prediction = (tf.nn.softmax(self.c_nuggets_logits, axis=-1),
                               tf.nn.softmax(self.h_nuggets_logits, axis=-1))

        elif task == Task.quality:
            self.quality_logits = quality_model_fn(
                features, self.dropout, params)
            self.loss = quality_loss(self.quality_logits, self.quality_labels)
            self.prediction = (tf.nn.softmax(self.quality_logits, axis=-1))

        else:
            raise ValueError("Unexpected Task: %s" % task.name)

        self.train_op = build_train_op(self.loss, tf.train.get_or_create_global_step(),
                                       lr=params.learning_rate, optimizer=params.optimizer)

    def save_model(self, save_path: Path = None):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.saver.save(self.session, str(save_path),
                        global_step=self.global_step)

    def load_model(self, restore_path=None, global_step=None):
        if not global_step:
            path = tf.train.latest_checkpoint(str(restore_path))
            self.saver.restore(self.session, path)
        else:
            self.saver.restore(self.session, path, global_step=global_step)

    def train_batch(self, batch_op):
        (_, turns, senders, turn_lengths, dialog_lengths,
         c_nugget_labels, h_nugget_labels, quality_labels) = self.session.run(batch_op)

        feed_dict = {
            self.turns: turns,
            self.senders: senders,
            self.turn_lengths: turn_lengths,
            self.dialogue_lengths: dialog_lengths,
            self.c_nuggets_labels: c_nugget_labels,
            self.h_nuggets_labels: h_nugget_labels,
            self.quality_labels: quality_labels,
        }

        _, loss = self.session.run(
            [self.train_op, self.loss],
            feed_dict=feed_dict,
            run_metadata=self.run_metadata,
            options=self.run_options
        )

        return loss

    def train_epoch(self, batch_initializer, batch_op, n_epoch=1,
                    reduce_fn=np.mean, save_path=None, save_per_epoch=True):
        results = []
        for i in range(n_epoch):
            self.session.run(batch_initializer)
            while True:
                try:
                    results.append(self.train_batch(batch_op))
                    self.global_step += 1
                except tf.errors.OutOfRangeError:
                    break
            if save_per_epoch and save_path:
                self.save_model(save_path)
        return reduce_fn(results)

    def __predict_batch(self, batch_op):
        (dialog_ids, turns, senders, turn_lengths,
         dialog_lengths) = self.session.run(batch_op)

        feed_dict = {
            self.turns: turns,
            self.senders: senders,
            self.turn_lengths: turn_lengths,
            self.dialogue_lengths: dialog_lengths,
            self.dropout: 0.,
        }

        outputs = self.session.run(self.prediction, feed_dict=feed_dict)

        if isinstance(outputs, tuple):
            return dialog_ids, zip(*outputs), dialog_lengths
        return dialog_ids, outputs, dialog_lengths

    def predict(self, batch_initializer, batch_op):
        results = []
        self.session.run(batch_initializer)
        while True:
            try:
                results.extend(zip(*self.__predict_batch(batch_op)))
            except tf.errors.OutOfRangeError:
                break
        return results


def _sender_aware_encoding(inputs, senders):
    """ convert embedding into sender-aware embedding
    if sender is 1, the embedding (the output) is [0, 0, ...,  0, embed[0], embed[1],....]
    if sender is 0, the embedding is [embed[0], embed[1], ..., 0, 0, 0]
    """
    with tf.name_scope("sender_aware_encoding"):
        inputs_repeat = tf.tile(inputs, [1, 1, 2])
        mask_0 = tf.tile(tf.expand_dims(tf.logical_not(
            senders), -1), [1, 1, tf.shape(inputs)[-1]])
        mask_1 = tf.tile(tf.expand_dims(senders, -1),
                         [1, 1, tf.shape(inputs)[-1]])
        mask = tf.concat([mask_0, mask_1], axis=-1)
        output = tf.where(mask, inputs_repeat, tf.zeros_like(inputs_repeat))
        return output


def _rnn(inputs, seq_lengths, dropout, params, name_scope="rnn"):
    # with tf.name_scope(name_scope):
    with tf.variable_scope(name_scope):
        def cell_fn(): return rnn_cell.DropoutWrapper(
            params.cell(params.hidden_size),
            output_keep_prob=1 - dropout,
            variational_recurrent=True,
            dtype=tf.float32
        )

        fw_cells = [cell_fn() for _ in range(params.num_layers)]
        bw_cells = [cell_fn() for _ in range(params.num_layers)]

        output, _, _ = stack_bidirectional_dynamic_rnn(fw_cells, bw_cells, inputs, sequence_length=seq_lengths,
                                                       dtype=tf.float32)

        return output


def _encoder(inputs, senders, dialog_lengths, dropout, params):
    # [batch_num, dialog_len, (2 x embedding_size)]
    inputs = _sender_aware_encoding(inputs, senders)
    return _rnn(inputs, dialog_lengths, dropout, params)


def quality_model_fn(features, dropout, params):
    turns, senders, utterance_lengths, dialog_lengths = features
    output = _encoder(turns, senders, dialog_lengths, dropout, params)
    dialog_repr = tf.reduce_sum(output, axis=1)
    logits = []
    for _ in data.QUALITY_MEASURES:
        logits.append(tf.layers.dense(dialog_repr, len(data.QUALITY_SCALES)))
    logits = tf.stack(logits, axis=1)
    return logits


def nugget_model_fn(features, dropout, params):
    turns, senders, utterance_lengths, dialog_lengths = features
    output = _encoder(turns, senders, dialog_lengths, dropout, params)

    # assume ordering is  [customer, helpdesk, customer, .....]
    max_time = tf.shape(output)[1]
    customer_index = tf.range(start=0, delta=2, limit=max_time)
    helpdesk_index = tf.range(start=1, delta=2, limit=max_time)

    customer_output = tf.gather(output, indices=customer_index, axis=1)
    helpdesk_output = tf.gather(output, indices=helpdesk_index, axis=1)

    assert_op = tf.assert_equal(tf.shape(customer_output)[
                                1] + tf.shape(helpdesk_output)[1], max_time)

    with tf.control_dependencies([assert_op]):
        customer_logits = tf.layers.dense(
            customer_output, len(data.CUSTOMER_NUGGET_TYPES_WITH_PAD))
        helpdesk_logits = tf.layers.dense(
            helpdesk_output, len(data.HELPDESK_NUGGET_TYPES_WITH_PAD))

    return customer_logits, helpdesk_logits


def nugget_loss(customer_logits, helpdesk_logits, customer_labels, helpdesk_labels, dialogue_lengths, max_dialogue_len):
    mask = tf.sequence_mask(dialogue_lengths)

    customer_index = tf.range(start=0, delta=2, limit=max_dialogue_len)
    helpdesk_index = tf.range(start=1, delta=2, limit=max_dialogue_len)

    customer_mask = tf.gather(mask, indices=customer_index, axis=1)
    helpdesk_mask = tf.gather(mask, indices=helpdesk_index, axis=1)

    customer_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=customer_logits, labels=customer_labels)
    customer_loss = tf.reduce_sum(tf.where(customer_mask, customer_loss, tf.zeros_like(customer_loss))) \
        / tf.cast(tf.shape(customer_logits)[0], dtype=tf.float32)

    helpdesk_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=helpdesk_logits, labels=helpdesk_labels)
    helpdesk_loss = tf.reduce_sum(tf.where(helpdesk_mask, helpdesk_loss, tf.zeros_like(helpdesk_loss))) \
        / tf.cast(tf.shape(helpdesk_logits)[0], dtype=tf.float32)

    return helpdesk_loss + customer_loss


def quality_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))


def build_train_op(loss, global_step, optimizer=None, lr=None, moving_decay=0.9999):
    if lr is not None:
        opt = optimizer(lr)
    else:
        opt = optimizer()
    grads = opt.compute_gradients(loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op


class DiffBinModel(Model):

    def _set_operations(self, embedding, task, params):
        self.is_train = params.is_train

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(shape=embedding.shape, trainable=params.update_embedding,
                                             name="embedding_weights", dtype=tf.float32)
            turns_embedded = tf.nn.embedding_lookup(self.embedding, self.turns)
        turns_boW = tf.reduce_sum(
            turns_embedded, axis=2, name="BoW")  # Bag of Words

        features = (turns_boW, self.senders,
                    self.turn_lengths, self.dialogue_lengths)

        if task == Task.nugget:
            raise Exception("DiffBinModel not implemented for nugget task")

        elif task == Task.quality:
            self.quality_logits = quality_model_fn(
                features, self.dropout, params)
            dist_loss = quality_loss(self.quality_logits, self.quality_labels)
            self.prediction = (tf.nn.softmax(self.quality_logits, axis=-1))

            left_prediction, _ = tf.split(self.prediction, [4, 1], 2)
            _, right_prediction = tf.split(self.prediction, [1, 4], 2)
            left_quality_labels, _ = tf.split(self.quality_labels, [4, 1], 2)
            _, right_quality_labels = tf.split(self.quality_labels, [1, 4], 2)
            diff_prediction = tf.subtract(
                left_prediction, right_prediction)
            diff_quality_labels = tf.subtract(
                left_quality_labels, right_quality_labels)
            diff_loss = tf.losses.mean_squared_error(
                diff_prediction, diff_quality_labels)

            alpha = tf.constant(params.alpha)
            self.loss = alpha * dist_loss + (1 - alpha) * diff_loss

        else:
            raise ValueError("Unexpected Task: %s" % task.name)

        self.train_op = build_train_op(self.loss, tf.train.get_or_create_global_step(),
                                       lr=params.learning_rate, optimizer=params.optimizer)


class AMTLModel(Model):

    def _set_operations(self, embedding, task, params):
        self.is_train = params.is_train

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(shape=embedding.shape, trainable=params.update_embedding,
                                             name="embedding_weights", dtype=tf.float32)
            turns_embedded = tf.nn.embedding_lookup(self.embedding, self.turns)
        turns_boW = tf.reduce_sum(
            turns_embedded, axis=2, name="BoW")  # Bag of Words

        inputs = _sender_aware_encoding(turns_boW, self.senders)
        # quality_output = _rnn(inputs, self.dialogue_lengths,
        #                       self.dropout, params, name_scope="quality_rnn")
        # nugget_output = _rnn(inputs, self.dialogue_lengths,
        #                      self.dropout, params, name_scope="nugget_rnn")

        with tf.name_scope("shared_rnn"):
            def cell_fn(): return rnn_cell.DropoutWrapper(
                params.cell(params.hidden_size),
                output_keep_prob=1 - self.dropout,
                variational_recurrent=True,
                dtype=tf.float32
            )

            shared_fw_cells = [cell_fn() for _ in range(params.num_layers)]
            shared_bw_cells = [cell_fn() for _ in range(params.num_layers)]

            shared_output, _, _ = stack_bidirectional_dynamic_rnn(
                shared_fw_cells,
                shared_bw_cells,
                inputs,
                sequence_length=self.dialogue_lengths,
                dtype=tf.float32)

        # quality_shared_repr = tf.concat(
        #     [quality_output, shared_output], axis=2)
        # nugget_shared_repr = tf.concat([nugget_output, shared_output], axis=2)

        # self.shared_dense = tf.layers.Dense(units=2, activation=None)

        # Quality task loss
        # quality_dialogue_repr = tf.reduce_sum(quality_shared_repr, axis=1)
        quality_dialogue_repr = tf.reduce_sum(shared_output, axis=1)
        quality_logits = []
        for _ in data.QUALITY_MEASURES:
            quality_logits.append(tf.layers.dense(
                quality_dialogue_repr, len(data.QUALITY_SCALES)))
        quality_logits = tf.stack(quality_logits, axis=1)
        dist_loss = quality_loss(quality_logits, self.quality_labels)
        self.quality_prediction = (tf.nn.softmax(quality_logits, axis=-1))

        left_prediction, _ = tf.split(self.quality_prediction, [4, 1], 2)
        _, right_prediction = tf.split(self.quality_prediction, [1, 4], 2)
        left_quality_labels, _ = tf.split(self.quality_labels, [4, 1], 2)
        _, right_quality_labels = tf.split(self.quality_labels, [1, 4], 2)
        diff_prediction = tf.subtract(
            left_prediction, right_prediction)
        diff_quality_labels = tf.subtract(
            left_quality_labels, right_quality_labels)
        diff_loss = tf.losses.mean_squared_error(
            diff_prediction, diff_quality_labels)

        alpha = tf.constant(params.alpha)
        self.quality_loss = alpha * dist_loss + (1 - alpha) * diff_loss

        # Nugget Task loss
        nugget_shared_repr = shared_output  # not use
        max_time = tf.shape(nugget_shared_repr)[1]
        customer_index = tf.range(start=0, delta=2, limit=max_time)
        helpdesk_index = tf.range(start=1, delta=2, limit=max_time)

        customer_output = tf.gather(
            nugget_shared_repr, indices=customer_index, axis=1)
        helpdesk_output = tf.gather(
            nugget_shared_repr, indices=helpdesk_index, axis=1)

        assert_op = tf.assert_equal(tf.shape(customer_output)[
                                    1] + tf.shape(helpdesk_output)[1], max_time)

        with tf.control_dependencies([assert_op]):
            self.c_nuggets_logits = tf.layers.dense(
                customer_output, len(data.CUSTOMER_NUGGET_TYPES_WITH_PAD))
            self.h_nuggets_logits = tf.layers.dense(
                helpdesk_output, len(data.HELPDESK_NUGGET_TYPES_WITH_PAD))

        self.nugget_loss = nugget_loss(
            self.c_nuggets_logits, self.h_nuggets_logits,
            self.c_nuggets_labels, self.h_nuggets_labels, self.dialogue_lengths, tf.shape(self.turns)[1])

        self.nugget_prediction = (tf.nn.softmax(self.c_nuggets_logits, axis=-1),
                                  tf.nn.softmax(self.h_nuggets_logits, axis=-1))

        # Train operations
        # shared_dialogue_repr = tf.reduce_sum(shared_output, axis=1)
        # nugget_dialogue_repr = tf.reduce_sum(nugget_shared_repr, axis=1)
        # loss_diff = self.diff_loss(nugget_dialogue_repr, quality_dialogue_repr)
        self.loss = []
        self.train_op = []
        for task in ["quality", "nugget"]:
            # loss_adv, loss_adv_l2 = self.adversarial_loss(
            #     shared_dialogue_repr, Task[task].value, self.dropout)
            # task_loss = getattr(self,  "%s_loss" % task) + \
            #     0.05 * loss_adv + loss_diff
            task_loss = getattr(self,  "%s_loss" % task)
            self.loss.append(task_loss)
            self.train_op.append(build_train_op(task_loss, tf.train.get_or_create_global_step(),
                                                lr=params.learning_rate, optimizer=params.optimizer))

        # Prections
        # self.prediction = self.quality_prediction + self.nugget_prediction

    def diff_loss(self, shared_feat, task_feat):
        task_feat -= tf.reduce_mean(task_feat, 0)
        shared_feat -= tf.reduce_mean(shared_feat, 0)

        task_feat = tf.nn.l2_normalize(task_feat, 1)
        shared_feat = tf.nn.l2_normalize(shared_feat, 1)

        correlation_matrix = tf.matmul(
            task_feat, shared_feat, transpose_a=True)

        cost = tf.reduce_mean(tf.square(correlation_matrix)) * 0.01
        cost = tf.where(cost > 0, cost, 0, name='value')

        assert_op = tf.Assert(tf.is_finite(cost), [cost])
        with tf.control_dependencies([assert_op]):
            loss_diff = tf.identity(cost)

        return loss_diff

    def adversarial_loss(self, repr, task_label, dropout, is_train=True):
        repr = flip_gradient(repr)
        if self.is_train:
            repr = tf.nn.dropout(repr, dropout)

        # shared_dense = tf.layers.Dense(units=2, activation=None)
        shared_logits = self.shared_dense(inputs=repr)
        shared_dense_l2 = tf.nn.l2_loss(self.shared_dense.weights[0]) \
            + tf.nn.l2_loss(self.shared_dense.weights[1])
        # import ipdb
        # ipdb.set_trace()
        label = tf.one_hot(task_label, 2)
        loss_adv = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=shared_logits))

        return loss_adv, shared_dense_l2

    def train_batch(self, batch_op):
        (_, turns, senders, turn_lengths, dialog_lengths,
         c_nugget_labels, h_nugget_labels, quality_labels) = self.session.run(batch_op)

        feed_dict = {
            self.turns: turns,
            self.senders: senders,
            self.turn_lengths: turn_lengths,
            self.dialogue_lengths: dialog_lengths,
            self.c_nuggets_labels: c_nugget_labels,
            self.h_nuggets_labels: h_nugget_labels,
            self.quality_labels: quality_labels,
        }

        _, _, qloss,  nloss = self.session.run(
            (self.train_op + self.loss),
            feed_dict=feed_dict,
            run_metadata=self.run_metadata,
            options=self.run_options
        )

        return qloss, nloss


class FlipGradientBuilder(object):
    '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()
