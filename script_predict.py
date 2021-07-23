#!/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from model.mtl2_model import MTL2CharCNNWordBilstmModel
from model.mtl3_model import MTL3CharCNNWordBilstmModel
from model.mtl4_model import MTL4CharCNNWordBilstmModel
from model.stl_model import STLCharCNNWordBilstmModel
from common.charnomalizer import CharNormalizer
from common.config import Config
from common.utility import *
import os.path
import pickle
import io

from os import listdir
from os.path import isfile, join

model_type = sys.argv[1] #stl/mtl
model_path = sys.argv[2]
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]
file_full_word_embedding = "./files/mtl/ner_bijankhan/we.vec"

print("\n\n----------------start loading vocabs--------------\n\n")

cfg = Config("", [], model_path, file_full_word_embedding)

[vocab_id2tag, vocab_tag2id] = load_vocab(cfg.file_tag_vocab)
[vocab_id2word, vocab_word2id] = load_vocab(cfg.file_word_vocab)
[vocab_id2char, vocab_char2id] = load_vocab(cfg.file_char_vocab)



print("\n\n-------------------start load and check word embedding-----------------\n\n")

if not os.path.isfile(cfg.file_full_word_embedding + '.pickle'):
    word_embedding, avg_vector = get_full_word_embeddings(cfg.file_full_word_embedding, cfg.word_embedding_dimension)
    with open(cfg.file_full_word_embedding + '.pickle', 'wb') as handle:
        pickle.dump([word_embedding, avg_vector], handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(cfg.file_full_word_embedding + '.pickle', 'rb') as f:
        [word_embedding, avg_vector] = pickle.load(f)

twe = load_trimmed_word_embeddings(cfg.file_trimmed_word_embedding)
[vocab_size, dim] = np.shape(twe)

tag_size = max(vocab_id2tag, key=int) + 1
char_size = max(vocab_id2char, key=int) + 1

normalizer = CharNormalizer()


if model_type == "stl":
    mod = STLCharCNNWordBilstmModel(vocab_size, dim, tag_size, cfg.max_char, cfg.char_embedding_dimension,
                                    cfg.wrd_lstm_hidden_size
                                    , cfg.learning_rate, cfg.dir_tensoboard_log, cfg.dir_checkpoints, char_size)
elif model_type == "mtl2":
    mod = MTL2CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, cfg.max_char, cfg.char_embedding_dimension,
                                     cfg.wrd_lstm_hidden_size
                                     , cfg.learning_rate, cfg.dir_tensoboard_log, cfg.dir_checkpoints, char_size)
elif model_type == "mtl3":
    mod = MTL3CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, tag_size, cfg.max_char, cfg.char_embedding_dimension,
                                     cfg.wrd_lstm_hidden_size
                                     , cfg.learning_rate, cfg.dir_tensoboard_log, cfg.dir_checkpoints, char_size)
elif model_type == "mtl4":
    mod = MTL4CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, tag_size, tag_size, cfg.max_char, cfg.char_embedding_dimension,
                                     cfg.wrd_lstm_hidden_size
                                     , cfg.learning_rate, cfg.dir_tensoboard_log, cfg.dir_checkpoints, char_size)

mod.build_graph()
mod.restore_graph()

vocab_id2word[-1] = "OOV"

with io.open(input_file_path, 'r', encoding='utf-8') as input_file:
  lines = input_file.readlines()
  for line in lines:
    sen = line.split()

    w_ids = []
    ch_ids = []

    print("\n\n---------------convert word to word_ids and char_ids----------------\n\n")

    for word in sen:
      word = word.encode('utf-8')
      w_id, twe = update_word_vocab(word, vocab_id2word, vocab_word2id, twe, word_embedding, avg_vector)
      w_ids.append(w_id)

      word_chars = [vocab_char2id[x] for x in word if x in vocab_char2id.keys()]
      word_chars += [0] * (cfg.max_char - len(word_chars))

      ch_ids.append(word_chars)

      word_ids = []
      word_ids.append(w_ids)
      char_ids = []
      char_ids.append(ch_ids)

      words, sen_len = pad_sequences(word_ids, 0)
      chars, word_len = pad_sequences(char_ids, 0, nlevels=2)

    labels = mod.sess.run(mod.labels_pred, feed_dict={mod.word_ids:word_ids, mod.char_ids:char_ids, mod.sentence_lenghts:sen_len, mod.word_lengths:word_len, mod.word_embeddings:twe, mod.dropout:1.0})
    with io.open(output_file_path, 'w', encoding='utf-8') as output_file:
      for ind,_ in enumerate(sen):
        output_file.write(u"{} {}\n".format(sen[ind], vocab_id2tag[labels[0,ind]]))
