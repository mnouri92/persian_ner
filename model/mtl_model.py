import tensorflow as tf
import numpy as np
from common.utility import setup_custom_logger
from common.utility import remove_padding, pad_sequences
import os
from sklearn.metrics import confusion_matrix

class MTL2CharCNNWordBilstmModel():

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

    def build_graph(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_lstm()
        self.add_fcn()
        self.add_train_op()
        self.add_pred_op()
        self.initialize_session()
        self.merged = tf.summary.merge_all()

    def add_placeholders(self):

        # [batch_size, max_sentence_length_in_batch]
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_ids')

        # [batch_size]
        self.sentence_lenghts = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence_lenghts')

        # [batch_size, max_sentence_length_in_batch, max_word_length_in_batch]
        self.char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, self.max_word_len], name='char_ids')

        # [batch_size, max_sentence_length_in_batch]
        self.word_lengths = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_lenghts')

        # [batch_size, max_sentence_length_in_batch]
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name='labels')

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')

    def add_word_embeddings_op(self):

        # [vocab_size, embedding_dim]
        self.word_embeddings = tf.placeholder(dtype=tf.float32, shape=[None, self.dim])
        # v = tf.Variable(dtype=tf.float32, initial_value=np.full((self.vocab_size, self.dim), 0))
        # self.word_embeddings_var = tf.assign(v, self.word_embeddings)
        # self.word_embeddings_var = tf.get_variable(name="word_embeddings", dtype=tf.float32 , shape=[self.vocab_size, self.dim], initializer=tf.random_normal_initializer(), trainable=True)
        self.embedded_words = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids, name='embedded_words')

        char_embedding = tf.get_variable(name="char_embeddings", dtype=tf.float32
                                         , shape=[self.char_size, self.char_emb_dim])
        embedded_chars = tf.nn.embedding_lookup(char_embedding, self.char_ids, name='embedded_chars')

        s = tf.shape(embedded_chars)

        embedded_chars = tf.reshape(embedded_chars, shape=[-1,self.max_word_len,self.char_emb_dim])

        embedded_chars = tf.expand_dims(embedded_chars, -1)

        num_filter = 128
        filter_sizes= [2,3,4,5,6]
        pooled_outputs = []

        for filter_size in filter_sizes:
            conv = tf.layers.conv2d(embedded_chars, num_filter, (filter_size, self.char_emb_dim),
                                     activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(conv, (self.max_word_len - filter_size + 1, 1), (1, 1))
            pool = tf.reshape(pool, shape=[s[0], s[1], num_filter])
            pooled_outputs.append(pool)
        concat_pooled = tf.concat(pooled_outputs, 2)


        self.embedded_words = tf.concat([self.embedded_words, concat_pooled], axis=-1)
        self.embedded_words = tf.nn.dropout(self.embedded_words, self.dropout)

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
            self.task1_loss = tf.reduce_mean(-log_likelihood)
            self.task1_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.task1_loss)
            self.task1_trainloss = tf.summary.scalar('task1 train batch loss', self.task1_loss)
            self.task1_validationloss = tf.summary.scalar('task1 validation loss', self.task1_loss)

        with tf.variable_scope('task2_loss'):
            log_likelihood, self.task2_transition_param = tf.contrib.crf.crf_log_likelihood(self.task2_logits, self.labels, self.sentence_lenghts)
            self.task2_loss = tf.reduce_mean(-log_likelihood)
            self.task2_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.task2_loss)
            self.task2_trainloss = tf.summary.scalar('task2 train batch loss', self.task2_loss)
            self.task2_validationloss = tf.summary.scalar('task2 validation loss', self.task2_loss)

    def add_pred_op(self):

        self.task1_labels_pred = tf.contrib.crf.crf_decode(self.task1_logits, self.task1_transition_param, self.sentence_lenghts)[0]
        print(self.task1_labels_pred)

        self.task2_labels_pred = tf.contrib.crf.crf_decode(self.task2_logits, self.task2_transition_param, self.sentence_lenghts)[0]
        print(self.task2_labels_pred)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)
        self.writer = tf.summary.FileWriter(self.tensorboard_log, graph=tf.get_default_graph())

    def create_feed_dict(self, word_seq, tag_seq, char_seq, word_embedding, start_index, end_index, dropout):

        current_batch_word_seq = word_seq[start_index:end_index]
        current_batch_char_seq = char_seq[start_index:end_index]
        current_batch_tag_seq = tag_seq[start_index:end_index]

        current_batch_word_seq, current_batch_sen_len = pad_sequences(current_batch_word_seq, 0)
        current_batch_tag_seq, _ = pad_sequences(current_batch_tag_seq, 0)
        current_batch_char_seq, current_batch_word_len = pad_sequences(current_batch_char_seq, 0, nlevels=2)

        feed_dict = {
            self.word_ids: current_batch_word_seq,
            self.labels: current_batch_tag_seq,
            self.sentence_lenghts: current_batch_sen_len,
            self.word_embeddings: word_embedding,
            self.dropout: dropout,
            self.char_ids: current_batch_char_seq,
            self.word_lengths: current_batch_word_len
        }
        return feed_dict, current_batch_sen_len, current_batch_word_seq, current_batch_tag_seq

    def train_graph(self, main_task_train_word_seq, main_task_train_tag_seq, main_task_train_char_seq,
                    aux_task_train_word_seq, aux_task_train_tag_seq, aux_task_train_char_seq,
                    val_word_seq, val_tag_seq, val_char_seq,
                    word_embedding, epoch_start, epoch_end, batch_size):

        task1_num_sen = len(main_task_train_word_seq)
        task2_num_sen = len(aux_task_train_word_seq)

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
                    self.create_feed_dict(main_task_train_word_seq, main_task_train_tag_seq, main_task_train_char_seq, word_embedding, start_index_task1, end_index_task1, self.dropout)
                [summary, _, loss] = self.sess.run([self.task1_trainloss, self.task1_train, self.task1_loss], feed_dict)
                if batch_number_task1 % 50 == 0:
                    self.writer.add_summary(summary, total_counter)
                    self.logger.info("epoch: {} batch: {} task: 1 loss on train: {}".format(epoch, batch_number_task1, loss))

                feed_dict, current_batch_len, current_batch_word_seq, current_batch_tag_seq = \
                    self.create_feed_dict(aux_task_train_word_seq, aux_task_train_tag_seq, aux_task_train_char_seq, word_embedding, start_index_task2, end_index_task2, self.dropout)
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


    def restore_graph(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.chkpnts_path))
        return tf.train.latest_checkpoint(self.chkpnts_path)

    def evaluate_model(self, test_word_seq, test_tag_seq, test_char_seq, word_embedding, batch_size, task_number=1, id2word={}, id2tag={},
                       result_file_path=''):
        try:
            os.remove(result_file_path)
        except OSError:
            pass

        total_num_sentences = np.shape(test_word_seq)[0]
        start_index = 0
        total_loss = 0
        all_predicted = []
        all_label = []
        batch_number = 0
        start_index = 0
        end_index = 0

        while end_index < total_num_sentences:
            start_index = batch_number * batch_size
            batch_number += 1
            end_index = min([total_num_sentences, start_index + batch_size])

            feed_dict, current_batch_test_sen_len, current_batch_test_word_seq, current_batch_test_tag_seq = \
                self.create_feed_dict(test_word_seq, test_tag_seq, test_char_seq, word_embedding, start_index,
                                      end_index, 1.0)

            predicted = 0
            if task_number==1:
                [predicted, loss] = self.sess.run([self.task1_labels_pred, self.task1_loss], feed_dict=feed_dict)
                total_loss += loss
            elif task_number==2:
                [predicted, loss] = self.sess.run([self.task2_labels_pred, self.task2_loss], feed_dict=feed_dict)
                total_loss += loss


            non_padded_predicted = remove_padding(predicted, current_batch_test_sen_len)
            all_predicted += non_padded_predicted
            non_padded_label = remove_padding(current_batch_test_tag_seq, current_batch_test_sen_len)
            all_label += non_padded_label
            non_padded_word = remove_padding(current_batch_test_word_seq, current_batch_test_sen_len)

            num_of_sentence = len(non_padded_word)

            if result_file_path:
                with open(result_file_path, 'a') as f:
                    for i in range(num_of_sentence):
                        for j in range(current_batch_test_sen_len[i]):
                            word_index = non_padded_word[i][j]
                            label_index = non_padded_label[i][j]
                            predicted_index = non_padded_predicted[i][j]

                            # f.write(
                            #     '{}\tPOS\t{}\t{}\n'.format(id2word[word_index], id2tag[label_index],
                            #                                id2tag[predicted_index]))
                            f.write('{}\n'.format(id2tag[predicted_index]))
                        f.write('\n')

        flat_pred = []
        [flat_pred.extend(p) for p in all_predicted]

        flat_orig = []
        [flat_orig.extend(p) for p in all_label]

        np.set_printoptions(linewidth=200)
        cm = confusion_matrix(flat_pred, flat_orig)
        acc = np.sum(np.diag(cm[:-1, :-1])) / np.sum(cm[:, :-1])

        return acc


    def extract_ner_tags(self, test_word_seq, test_char_seq, word_embedding, task_num):
        total_num_sentences = np.shape(test_word_seq)[0]
        batch_size = 256
        start_index = 0
        all_predicted = []

        while (start_index < total_num_sentences):
            test_tag_seq = np.array([np.zeros_like(a) for a in test_word_seq])

            end_index = min([total_num_sentences, start_index + batch_size])

            feed_dict, current_batch_test_sen_len, current_batch_test_word_seq, current_batch_test_tag_seq = \
                self.create_feed_dict(test_word_seq, test_tag_seq, test_char_seq, word_embedding, start_index, end_index, 1.0)

            if(task_num == 1):
                [predicted, loss] = self.sess.run([self.task1_labels_pred, self.task1_loss], feed_dict=feed_dict)
            elif(task_num == 2):
                [predicted, loss] = self.sess.run([self.task2_labels_pred, self.task2_loss], feed_dict=feed_dict)

            non_padded_predicted = remove_padding(predicted, current_batch_test_sen_len)
            all_predicted += non_padded_predicted
            start_index = end_index

        return all_predicted






