from model.mtl2_model import MTL2CharCNNWordBilstmModel
from model.mtl3_model import MTL3CharCNNWordBilstmModel
from model.mtl4_model import MTL4CharCNNWordBilstmModel
from model.stl_model import STLCharCNNWordBilstmModel
from common.config import Config
from common.utility import *
import os.path
import pickle
import tensorflow as tf
from os import listdir
from os.path import isfile, join

type = sys.argv[1] #stl/mtl
model_path = sys.argv[2]
file_full_word_embedding =sys.argv[3]
export_path = sys.argv[4]
export_dir = os.path.join(export_path, '1')

cfg = Config("", [], model_path, file_full_word_embedding)


[vocab_id2tag, vocab_tag2id] = load_vocab(cfg.file_tag_vocab)
[vocab_id2word, vocab_word2id] = load_vocab(cfg.file_word_vocab)
[vocab_id2char, vocab_char2id] = load_vocab(cfg.file_char_vocab)

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

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
words_inputs_info = tf.saved_model.utils.build_tensor_info(mod.word_ids)
chars_inputs_info = tf.saved_model.utils.build_tensor_info(mod.char_ids)
word_len_info = tf.saved_model.utils.build_tensor_info(mod.word_lengths)
sen_len_info = tf.saved_model.utils.build_tensor_info(mod.sentence_lenghts)
word_embedding_inputs_info = tf.saved_model.utils.build_tensor_info(mod.word_embeddings)
dropout_inputs_info = tf.saved_model.utils.build_tensor_info(mod.dropout)
outputs_info = tf.saved_model.utils.build_tensor_info(mod.labels_pred)



prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'word_ids': words_inputs_info, 'char_ids': chars_inputs_info, 'word_len':word_len_info, 'sen_len':sen_len_info, 'word_embeddings':word_embedding_inputs_info, 'dropout':dropout_inputs_info},
                    outputs={'labels': outputs_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
                mod.sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'serving_default': prediction_signature
                },
                legacy_init_op=legacy_init_op)

builder.save()
                                                      
