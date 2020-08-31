import tensorflow as tf
import numpy as np
from common.utility import remove_padding, pad_sequences
import os
from sklearn.metrics import confusion_matrix
import os.path

class Model(tf.keras.Model):

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

        self.word_ids           = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_ids')                       #shape=[batch_size, max_sentence_length_in_batch]
        self.sentence_lenghts   = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence_lenghts')                     #shape=[batch_size]
        self.char_ids           = tf.placeholder(dtype=tf.int32, shape=[None, None, self.max_word_len], name='char_ids')    #shape=[batch_size, max_sentence_length_in_batch, max_word_length_in_batch]
        self.word_lengths       = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_lenghts')                   #shape=[batch_size, max_sentence_length_in_batch]
        self.labels             = tf.placeholder(dtype=tf.int32, shape=[None, None], name='labels')                         #shape=[batch_size, max_sentence_length_in_batch]
        self.dropout            = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')                                #shape=scalar
        self.word_embeddings    = tf.placeholder(dtype=tf.float32, shape=[None, self.dim])                                  #shape=[vocab_size, embedding_dim]

    def add_word_embeddings_op(self):
        self.embedded_words = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids, name='embedded_words')            #shape=[batch_size, max_sentence_length_in_batch, word_emb_dim]
        char_embedding = tf.get_variable(name="char_embeddings", dtype=tf.float32
                                         , shape=[self.char_size, self.char_emb_dim])                                       #shape=[num_unique_chars, char_emb_dim]
        embedded_chars = tf.nn.embedding_lookup(char_embedding, self.char_ids, name='embedded_chars')                       #shape=[batch_size, max_sentence_length_in_batch, max_word_length_in_batch, char_emb_dim]
        s = tf.shape(embedded_chars)
        embedded_chars = tf.reshape(embedded_chars, shape=[-1,self.max_word_len,self.char_emb_dim])                         #shape=[batch_size*max_sentence_length_in_batch, max_word_length_in_batch, char_emb_dim]
        embedded_chars = tf.expand_dims(embedded_chars, -1)                                                                 #shape=[batch_size*max_sentence_length_in_batch, max_word_length_in_batch, char_emb_dim, 1]

        num_filter = 128
        filter_sizes= [2,3,4,5,6]
        pooled_outputs = []
        for filter_size in filter_sizes:
            conv = tf.layers.conv2d(embedded_chars, num_filter, (filter_size, self.char_emb_dim),
                                     activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(conv, (self.max_word_len - filter_size + 1, 1), (1, 1))
            pool = tf.reshape(pool, shape=[s[0], s[1], num_filter])
            pooled_outputs.append(pool)
        concat_pooled = tf.concat(pooled_outputs, 2)                                                                        #shape=[batch_size, max_sentence_length_in_batch, num_filter*len(filter_sizes)]

        self.embedded_words = tf.concat([self.embedded_words, concat_pooled], axis=-1)                                      #shape=[batch_size, max_sentence_length_in_batch, num_filter*len(filter_sizes)+word_emb_dim]
        self.embedded_words = tf.nn.dropout(self.embedded_words, self.dropout)                                              #shape=[batch_size, max_sentence_length_in_batch, num_filter*len(filter_sizes)+word_emb_dim]

    def add_pred_op(self):

        self.labels_pred = tf.contrib.crf.crf_decode(self.task1_logits, self.task1_transition_param, self.sentence_lenghts)[0]

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

    def restore_graph(self):
        file_name = os.path.normpath(tf.train.latest_checkpoint(self.chkpnts_path))
        print(file_name)
        self.saver.restore(self.sess, file_name)
        return file_name

    def evaluate_model(self, test_word_seq, test_tag_seq, test_char_seq, word_embedding, batch_size, id2word={}, id2tag={},
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
            [predicted, loss] = self.sess.run([self.labels_pred, self.loss], feed_dict=feed_dict)
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

                            f.write(
                                '{}\tPOS\t{}\t{}\n'.format(id2word[word_index], id2tag[label_index],
                                                           id2tag[predicted_index]))
                            # f.write('{}\n'.format(id2tag[predicted_index]))
                        f.write('\n')

        flat_pred = []
        [flat_pred.extend(p) for p in all_predicted]

        flat_orig = []
        [flat_orig.extend(p) for p in all_label]

        np.set_printoptions(linewidth=200)
        cm = confusion_matrix(flat_pred, flat_orig)
        acc = np.sum(np.diag(cm[:-1, :-1])) / np.sum(cm[:, :-1])

        return acc


    def extract_ner_tags(self, test_word_seq, test_char_seq, word_embedding):
        total_num_sentences = np.shape(test_word_seq)[0]
        batch_size = 256
        start_index = 0
        all_predicted = []

        while (start_index < total_num_sentences):
            test_tag_seq = np.array([np.zeros_like(a) for a in test_word_seq])

            end_index = min([total_num_sentences, start_index + batch_size])

            feed_dict, current_batch_test_sen_len, current_batch_test_word_seq, current_batch_test_tag_seq = \
                self.create_feed_dict(test_word_seq, test_tag_seq, test_char_seq, word_embedding, start_index, end_index, 1.0)

            [predicted, loss] = self.sess.run([self.labels_pred, self.loss], feed_dict=feed_dict)

            non_padded_predicted = remove_padding(predicted, current_batch_test_sen_len)
            all_predicted += non_padded_predicted
            start_index = end_index

        return all_predicted

