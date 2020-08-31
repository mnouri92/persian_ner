from model.mtl2_model import MTL2CharCNNWordBilstmModel
from model.mtl3_model import MTL3CharCNNWordBilstmModel
from model.mtl4_model import MTL4CharCNNWordBilstmModel
from model.stl_model import STLCharCNNWordBilstmModel
from common.config import Config
from common.utility import *
import pickle
from os import path
import tensorflow as tf


type = sys.argv[1] #stl/mtl
model_path = sys.argv[2]

cfg = Config("", [], model_path, "")

[vocab_id2tag, vocab_tag2id] = load_vocab(cfg.file_tag_vocab)
[vocab_id2word, vocab_word2id] = load_vocab(cfg.file_word_vocab)
[vocab_id2char, vocab_char2id] = load_vocab(cfg.file_char_vocab)

twe = load_trimmed_word_embeddings(cfg.file_trimmed_word_embedding)
[vocab_size, dim] = np.shape(twe)

tag_size = max(vocab_id2tag, key=int) + 1
char_size = max(vocab_id2char, key=int) + 1

if type == "stl":
    mod = STLCharCNNWordBilstmModel(vocab_size, dim, tag_size, cfg.max_char, cfg.char_embedding_dimension,
                                    cfg.wrd_lstm_hidden_size
                                    , cfg.learning_rate, cfg.dir_tensoboard_log, cfg.dir_checkpoints, char_size)
elif type == "mtl2":
    mod = MTL2CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, cfg.max_char, cfg.char_embedding_dimension,
                                     cfg.wrd_lstm_hidden_size
                                     , cfg.learning_rate, cfg.dir_tensoboard_log, cfg.dir_checkpoints, char_size)
elif type == "mtl3":
    mod = MTL3CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, tag_size, cfg.max_char, cfg.char_embedding_dimension,
                                     cfg.wrd_lstm_hidden_size
                                     , cfg.learning_rate, cfg.dir_tensoboard_log, cfg.dir_checkpoints, char_size)
elif type == "mtl4":
    mod = MTL4CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, tag_size, tag_size, cfg.max_char, cfg.char_embedding_dimension,
                                     cfg.wrd_lstm_hidden_size
                                     , cfg.learning_rate, cfg.dir_tensoboard_log, cfg.dir_checkpoints, char_size)

mod.build_graph()
mod.restore_graph()

tf.saved_model.save(mod, "final")
