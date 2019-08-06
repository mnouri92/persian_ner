import tensorflow as tf
from common.utility import setup_custom_logger
from model.base_model import Model

class MTL2CharCNNWordBilstmModel(Model):

    def __init__(self, vocab_size, dim, task1_tag_size, task2_tag_size, max_word_len, char_emb_dim, lstm_size, learning_rate
                 , tensorboard_log, chkpnts_path, char_size):
        self.vocab_size = vocab_size
        self.dim = dim
        self.task1_tag_size = task1_tag_size
        self.task2_tag_size = task2_tag_size
        self.char_size = char_size
        self.logger = setup_custom_logger(__name__)
        self.max_word_len = max_word_len
        self.char_emb_dim = char_emb_dim
        self.lstm_size = lstm_size
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.chkpnts_path = chkpnts_path
        print('MTL2CharCNNWordBilstmModel')

        return

    def add_lstm(self):
        with tf.variable_scope('task1_bilstm'):
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size)
            (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.embedded_words,
                sequence_length=self.sentence_lenghts,
                dtype=tf.float32)
            task1_output_word = tf.concat([outputs_fw, outputs_bw], axis=2)

        with tf.variable_scope('task2_bilstm'):
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size)
            (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.embedded_words,
                sequence_length=self.sentence_lenghts,
                dtype=tf.float32)
            task2_output_word = tf.concat([outputs_fw, outputs_bw], axis=2)

        with tf.variable_scope('shared_bilstm'):
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size)
            (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.embedded_words,
                sequence_length=self.sentence_lenghts,
                dtype=tf.float32)
            shared_output_word = tf.concat([outputs_fw, outputs_bw], axis=2)

        task1_lstm_layer_output = tf.concat([task1_output_word, shared_output_word], axis = 2)
        self.task1_lstm_layer_output = tf.nn.dropout(task1_lstm_layer_output, self.dropout)

        task2_lstm_layer_output = tf.concat([task2_output_word, shared_output_word], axis = 2)
        self.task2_lstm_layer_output = tf.nn.dropout(task2_lstm_layer_output, self.dropout)

    def add_fcn(self):
        with tf.variable_scope('task1_fcn'):
            task1_W = tf.get_variable(name="task1_W", dtype=tf.float32, shape=[4 * self.lstm_size, self.task1_tag_size])
            task1_b = tf.get_variable(name="task1_b", dtype=tf.float32, shape=[self.task1_tag_size], initializer=tf.zeros_initializer())
            nsteps = tf.shape(self.task1_lstm_layer_output)[1]
            output = tf.reshape(self.task1_lstm_layer_output, shape=[-1, 4 * self.lstm_size])
            output = tf.matmul(output, task1_W) + task1_b
            self.task1_logits = tf.reshape(output, shape=[-1, nsteps, self.task1_tag_size])

        with tf.variable_scope('task2_fcn'):
            task2_W = tf.get_variable(name="task2_W", dtype=tf.float32, shape=[4 * self.lstm_size, self.task2_tag_size])
            task2_b = tf.get_variable(name="task2_b", dtype=tf.float32, shape=[self.task2_tag_size], initializer=tf.zeros_initializer())
            nsteps = tf.shape(self.task2_lstm_layer_output)[1]
            output = tf.reshape(self.task2_lstm_layer_output, shape=[-1, 4 * self.lstm_size])
            output = tf.matmul(output, task2_W) + task2_b
            self.task2_logits = tf.reshape(output, shape=[-1, nsteps, self.task2_tag_size])

    def add_train_op(self):

        with tf.variable_scope('task1_loss'):
            log_likelihood, self.task1_transition_param = tf.contrib.crf.crf_log_likelihood(self.task1_logits, self.labels, self.sentence_lenghts)
            self.loss = tf.reduce_mean(-log_likelihood)
            self.task1_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.task1_trainloss = tf.summary.scalar('task1 train batch loss', self.loss)
            self.task1_validationloss = tf.summary.scalar('task1 validation loss', self.loss)

        with tf.variable_scope('task2_loss'):
            log_likelihood, self.task2_transition_param = tf.contrib.crf.crf_log_likelihood(self.task2_logits, self.labels, self.sentence_lenghts)
            self.task2_loss = tf.reduce_mean(-log_likelihood)
            self.task2_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.task2_loss)
            self.task2_trainloss = tf.summary.scalar('task2 train batch loss', self.task2_loss)
            self.task2_validationloss = tf.summary.scalar('task2 validation loss', self.task2_loss)


    def train_graph(self, main_task_train_word_seq, main_task_train_tag_seq, main_task_train_char_seq,
                    aux_task1_train_word_seq, aux_task1_train_tag_seq, aux_task1_train_char_seq,
                    val_word_seq, val_tag_seq, val_char_seq,
                    word_embedding, epoch_start, epoch_end, batch_size):

        task1_num_sen = len(main_task_train_word_seq)
        task2_num_sen = len(aux_task1_train_word_seq)

        total_counter = 0
        batch_number_task1 = 0
        batch_number_task2 = 0
        end_index_task1 = 0
        end_index_task2 = 0
        for epoch in range(epoch_start, epoch_end):
            batch_number_task1 = 0
            end_index_task1 = 0
            while end_index_task1 < task1_num_sen:
                total_counter += 1

                if end_index_task2 == task2_num_sen:
                    end_index_task2 = 0
                    batch_number_task2 = 0


                start_index_task1 = batch_number_task1 * batch_size
                end_index_task1 = min([start_index_task1 + batch_size, task1_num_sen])

                start_index_task2 = batch_number_task2 * batch_size
                end_index_task2 = min([start_index_task2 + batch_size, task2_num_sen])

                feed_dict, current_batch_len, current_batch_word_seq, current_batch_tag_seq = \
                    self.create_feed_dict(main_task_train_word_seq, main_task_train_tag_seq, main_task_train_char_seq, word_embedding, start_index_task1, end_index_task1, 0.5)
                [summary, _, loss] = self.sess.run([self.task1_trainloss, self.task1_train, self.task1_loss], feed_dict)
                if batch_number_task1 % 50 == 0:
                    self.writer.add_summary(summary, total_counter)
                    self.logger.info("epoch: {} batch: {} task: 1 loss on train: {}".format(epoch, batch_number_task1, loss))

                feed_dict, current_batch_len, current_batch_word_seq, current_batch_tag_seq = \
                    self.create_feed_dict(aux_task1_train_word_seq, aux_task1_train_tag_seq, aux_task1_train_char_seq, word_embedding, start_index_task2, end_index_task2, 0.5)
                [summary, _, loss] = self.sess.run([self.task2_trainloss, self.task2_train, self.task2_loss], feed_dict)
                if batch_number_task2 % 50 == 0:
                    self.writer.add_summary(summary, total_counter)
                    self.logger.info("epoch: {} batch: {} task: 2 loss on train: {}".format(epoch, batch_number_task2, loss))

                batch_number_task1 += 1
                batch_number_task2 += 1

            # choice1: save model after each epoch and terminate after specified epoch number
            save_path = self.saver.save(self.sess, "{}/bilstm_ner".format(self.chkpnts_path),
                                        global_step=int(epoch), write_meta_graph=False)
            self.logger.info("model is saved in: {}{}".format(save_path, ''.join([' '] * 100)))

            self.writer.add_summary(summary, epoch)
            acc = self.evaluate_model(val_word_seq, val_tag_seq, val_char_seq, word_embedding, batch_size)
            self.logger.info("epoch: {} accuracy on validation: {}".format(epoch, acc))

