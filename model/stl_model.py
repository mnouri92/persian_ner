import tensorflow as tf
import numpy as np
from common.utility import setup_custom_logger
from common.utility import remove_padding, pad_sequences
import os
from sklearn.metrics import confusion_matrix
from model.base_model import Model


class STLCharCNNWordBilstmModel(Model):

    def __init__(self, vocab_size, dim, tag_size, max_word_len, char_emb_dim, lstm_size, learning_rate
                 , tensorboard_log, chkpnts_path, char_size):
        self.vocab_size = vocab_size
        self.dim = dim
        self.tag_size = tag_size
        self.char_size = char_size
        self.logger = setup_custom_logger(__name__)
        self.max_word_len = max_word_len
        self.char_emb_dim = char_emb_dim
        self.lstm_size = lstm_size
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.chkpnts_path = chkpnts_path
        print('STLCharCNNWordBilstmModel')

        return

    def add_lstm(self):
        with tf.variable_scope('bilstm'):
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size)
            (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.embedded_words,
                sequence_length=self.sentence_lenghts,
                dtype=tf.float32)
            output_word = tf.concat([outputs_fw, outputs_bw], axis=2)


        self.lstm_layer_output = tf.nn.dropout(output_word, self.dropout)                                                   #shape=[batch_size, max_sentence_length_in_batch, 2*lstm_hidden_size]


    def add_fcn(self):
        with tf.variable_scope('fcn'):
            W = tf.get_variable(name="W", dtype=tf.float32, shape=[2 * self.lstm_size, self.tag_size])
            b = tf.get_variable(name="b", dtype=tf.float32, shape=[self.tag_size], initializer=tf.zeros_initializer())
            nsteps = tf.shape(self.lstm_layer_output)[1]
            output = tf.reshape(self.lstm_layer_output, shape=[-1, 2 * self.lstm_size])
            output = tf.matmul(output, W) + b
            self.logits = tf.reshape(output, shape=[-1, nsteps, self.tag_size])                                             #shape=[batch_size, max_sentence_length_in_batch, tag_size]

    def add_train_op(self):

        with tf.variable_scope('loss'):
            log_likelihood, self.transition_param = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.sentence_lenghts)
            self.loss = tf.reduce_mean(-log_likelihood)
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.trainloss = tf.summary.scalar('train batch loss', self.loss)
            self.validationloss = tf.summary.scalar('validation loss', self.loss)

    def add_pred_op(self):
        self.labels_pred = tf.contrib.crf.crf_decode(self.logits, self.transition_param, self.sentence_lenghts)[0]

    def train_graph(self, train_word_seq, train_tag_seq, train_char_seq, val_word_seq, val_tag_seq, val_char_seq,
                    word_embedding, epoch_start, epoch_end, batch_size):

        num_sen = len(train_word_seq)

        total_counter = 0
        batch_number = 0
        end_index = 0
        for epoch in range(epoch_start, epoch_end):
            batch_number = 0
            end_index = 0
            while end_index < num_sen:
                total_counter += 1

                start_index = batch_number * batch_size
                end_index = min([start_index + batch_size, num_sen])


                feed_dict, current_batch_len, current_batch_word_seq, current_batch_tag_seq = \
                    self.create_feed_dict(train_word_seq, train_tag_seq, train_char_seq, word_embedding
                                          , start_index, end_index, 0.5)
                [summary, _, loss] = self.sess.run([self.trainloss, self.train, self.loss], feed_dict)
                if batch_number % 50 == 0:
                    self.writer.add_summary(summary, total_counter)
                    self.logger.info("epoch: {} batch: {} loss on train: {}".format(epoch, batch_number, loss))

                batch_number += 1

            save_path = self.saver.save(self.sess, "{}/stl_ner".format(self.chkpnts_path),
                                        global_step=int(epoch), write_meta_graph=False)
            self.logger.info("model is saved in: {}{}".format(save_path, ''.join([' '] * 100)))

            self.writer.add_summary(summary, epoch)

            acc = self.evaluate_model(val_word_seq, val_tag_seq, val_char_seq, word_embedding, batch_size)
            self.logger.info("epoch: {} accuracy on validation: {}".format(epoch, acc))






