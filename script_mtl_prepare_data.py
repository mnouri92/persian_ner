from common.mtl_config import MTLConfig
from common.utility import *

logger = setup_custom_logger(__name__)
logger.info("==========================================================================================================")

config = MTLConfig(directory_data_param="/home/itrc/hadi/workspace/persian_ner/files/mtl/data/"
                             , file_conll_task1_directory_param="/home/itrc/hadi/workspace/persian_ner/files/mtl/data/ner_armanperso/"
                             , file_conll_task2_directory_param="/home/itrc/hadi/workspace/persian_ner/files/mtl/data/ner_bijankhan/"
                             , file_conll_task3_directory_param="/home/itrc/hadi/workspace/persian_ner/files/mtl/data/pos/"
                             , file_conll_task4_directory_param="/home/itrc/hadi/workspace/persian_ner/files/mtl/data/gen/"
                             , task_param="mtl4")



if config.task == "mtl2":
    logger.info("start a new session to run build_data (mtl2)...")
elif config.task == "mtl3":
    logger.info("start a new session to run build_data (mtl3)...")
elif config.task == "mtl4":
    logger.info("start a new session to run build_data (mtl4)...")

if config.task == "mtl2":
    [word_vocab, tag_vocab, char_vocab, max_char] = build_vocab(
        [config.file_conll_task1_train_data, config.file_conll_task1_test_data
            , config.file_conll_task2_train_data, config.file_conll_task2_test_data
            , config.file_conll_validaion_data
         ])
elif config.task == "mtl3":
    [word_vocab, tag_vocab, char_vocab, max_char] = build_vocab(
        [config.file_conll_task1_train_data, config.file_conll_task1_test_data, config.file_conll_task1_validation_data
            , config.file_conll_task2_train_data, config.file_conll_task2_test_data, config.file_conll_task2_validation_data
            , config.file_conll_task3_train_data, config.file_conll_task3_test_data, config.file_conll_task3_validation_data
         ])
elif config.task == "mtl4":
    [word_vocab, tag_vocab, char_vocab, max_char] = build_vocab(
        [config.file_conll_task1_train_data, config.file_conll_task1_test_data, config.file_conll_task1_validation_data
            , config.file_conll_task2_train_data, config.file_conll_task2_test_data, config.file_conll_task2_validation_data
            , config.file_conll_task3_train_data, config.file_conll_task3_test_data, config.file_conll_task3_validation_data
            , config.file_conll_task4_train_data, config.file_conll_task4_test_data, config.file_conll_task4_validation_data
         ])


dump_vocab(word_vocab, config.file_word_vocab)
dump_vocab(tag_vocab, config.file_tag_vocab)
dump_vocab(char_vocab, config.file_char_vocab)

[vocab_id2word, vocab_word2id] = load_vocab(config.file_word_vocab)
dump_trimmed_word_embeddings(vocab_word2id, config.file_full_word_embedding, config.file_trimmed_word_embedding,config.word_embedding_dimension)

[vocab_id2tag, vocab_tag2id] = load_vocab(config.file_tag_vocab)
[vocab_id2char, vocab_char2id] = load_vocab(config.file_char_vocab)

convert_conll_to_numpy_array(config.file_conll_task1_train_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             config.file_seq_task1_train_data, config.max_char)
convert_conll_to_numpy_array(config.file_conll_task1_test_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             config.file_seq_task1_test_data, config.max_char)
convert_conll_to_numpy_array(config.file_conll_task1_validation_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             config.file_seq_task1_validation_data, config.max_char)
convert_conll_to_numpy_array(config.file_conll_task2_train_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             config.file_seq_task2_train_data, config.max_char)
convert_conll_to_numpy_array(config.file_conll_task2_test_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             config.file_seq_task2_test_data, config.max_char)
convert_conll_to_numpy_array(config.file_conll_task2_validation_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             config.file_seq_task2_validation_data, config.max_char)
if config.task == "mtl3":
    convert_conll_to_numpy_array(config.file_conll_task3_train_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 config.file_seq_task3_train_data, config.max_char)
    convert_conll_to_numpy_array(config.file_conll_task3_test_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 config.file_seq_task3_test_data, config.max_char)
    convert_conll_to_numpy_array(config.file_conll_task3_validation_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 config.file_seq_task3_validation_data, config.max_char)
if config.task == "mtl4":
    convert_conll_to_numpy_array(config.file_conll_task3_train_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 config.file_seq_task3_train_data, config.max_char)
    convert_conll_to_numpy_array(config.file_conll_task3_test_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 config.file_seq_task3_test_data, config.max_char)
    convert_conll_to_numpy_array(config.file_conll_task3_validation_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 config.file_seq_task3_validation_data, config.max_char)
    convert_conll_to_numpy_array(config.file_conll_task4_train_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 config.file_seq_task4_train_data, config.max_char)
    convert_conll_to_numpy_array(config.file_conll_task4_test_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 config.file_seq_task4_test_data, config.max_char)
    convert_conll_to_numpy_array(config.file_conll_task4_validation_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 config.file_seq_task4_validation_data, config.max_char)
