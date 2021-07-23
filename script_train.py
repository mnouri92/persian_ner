from model.mtl2_model import MTL2CharCNNWordBilstmModel
from model.mtl3_model import MTL3CharCNNWordBilstmModel
from model.mtl4_model import MTL4CharCNNWordBilstmModel
from model.stl_model import STLCharCNNWordBilstmModel
from common.utility import *
from common.config import Config
import os.path

logger = setup_custom_logger(__name__)

type = sys.argv[1]
model_path = sys.argv[2]
file_full_word_embedding =sys.argv[3]
main_task_directory_path = sys.argv[4]
if type == "mtl2":
    aux_task1_directory_path = sys.argv[5]
elif type == "mtl3":
    aux_task1_directory_path = sys.argv[5]
    aux_task2_directory_path = sys.argv[6]
elif type == "mtl4":
    aux_task1_directory_path = sys.argv[5]
    aux_task2_directory_path = sys.argv[6]
    aux_task3_directory_path = sys.argv[7]

if type == "stl":
    cfg = Config(main_task_directory_path, [], model_path, file_full_word_embedding)
    all_files = [cfg.file_conll_main_task_train_data]
elif type == "mtl2":
    cfg = Config(main_task_directory_path,
                    [aux_task1_directory_path], model_path, file_full_word_embedding)
    all_files = cfg.file_conll_aux_task_train_data
    all_files.append(cfg.file_conll_main_task_train_data)
elif type == "mtl3":
    cfg = Config(main_task_directory_path,
                    [aux_task1_directory_path, aux_task2_directory_path], model_path,
                    file_full_word_embedding)
    all_files = cfg.file_conll_aux_task_train_data
    all_files.append(cfg.file_conll_main_task_train_data)
elif type == "mtl4":
    cfg = Config(main_task_directory_path, [aux_task1_directory_path, aux_task2_directory_path, aux_task3_directory_path], model_path, file_full_word_embedding)
    all_files = cfg.file_conll_aux_task_train_data
    all_files.append(cfg.file_conll_main_task_train_data)

answer = raw_input("continue from previous learned models (probabely to improve them)?(yes/no)")
if answer == "no":
    logger.info("bulding vocabulray")
    [word_vocab, tag_vocab, char_vocab, max_char] = build_vocab(all_files)

    logger.info("dumping created vocabulray")
    dump_vocab(word_vocab, cfg.file_word_vocab)
    dump_vocab(tag_vocab, cfg.file_tag_vocab)
    dump_vocab(char_vocab, cfg.file_char_vocab)

    logger.info("loading created vocabulray again")
    [vocab_id2word, vocab_word2id] = load_vocab(cfg.file_word_vocab)
    [vocab_id2tag, vocab_tag2id] = load_vocab(cfg.file_tag_vocab)
    [vocab_id2char, vocab_char2id] = load_vocab(cfg.file_char_vocab)

    logger.info("trimming word embedding")
    dump_trimmed_word_embeddings(vocab_word2id, cfg.file_full_word_embedding, cfg.file_trimmed_word_embedding,cfg.word_embedding_dimension)

[vocab_id2tag, vocab_tag2id] = load_vocab(cfg.file_tag_vocab)
[vocab_id2word, vocab_word2id] = load_vocab(cfg.file_word_vocab)
[vocab_id2char, vocab_char2id] = load_vocab(cfg.file_char_vocab)
twe = load_trimmed_word_embeddings(cfg.file_trimmed_word_embedding)

logger.info("converting to conll format")
convert_conll_to_numpy_array(cfg.file_conll_main_task_train_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             cfg.file_seq_main_task_train_data, cfg.max_char)
if type == "mtl2":
    if answer == "no" or not os.path.isfile(cfg.file_seq_aux_task_train_data[0] + ".npz"):
        convert_conll_to_numpy_array(cfg.file_conll_aux_task_train_data[0], vocab_word2id, vocab_tag2id, vocab_char2id,
                                 cfg.file_seq_aux_task_train_data[0], cfg.max_char)
elif type == "mtl3":
    if answer == "no" or not os.path.isfile(cfg.file_seq_aux_task_train_data[0] + ".npz"):
        convert_conll_to_numpy_array(cfg.file_conll_aux_task_train_data[0], vocab_word2id, vocab_tag2id, vocab_char2id,
                                 cfg.file_seq_aux_task_train_data[0], cfg.max_char)
    if answer == "no" or not os.path.isfile(cfg.file_seq_aux_task_train_data[1] + ".npz"):
        convert_conll_to_numpy_array(cfg.file_conll_aux_task_train_data[1], vocab_word2id, vocab_tag2id, vocab_char2id,
                                 cfg.file_seq_aux_task_train_data[1], cfg.max_char)
elif type == "mtl4":
    if answer == "no" or not os.path.isfile(cfg.file_seq_aux_task_train_data[0] + ".npz"):
        convert_conll_to_numpy_array(cfg.file_conll_aux_task_train_data[0], vocab_word2id, vocab_tag2id, vocab_char2id,
                                 cfg.file_seq_aux_task_train_data[0], cfg.max_char)
    if answer == "no" or not os.path.isfile(cfg.file_seq_aux_task_train_data[1] + ".npz"):
        convert_conll_to_numpy_array(cfg.file_conll_aux_task_train_data[1], vocab_word2id, vocab_tag2id, vocab_char2id,
                                 cfg.file_seq_aux_task_train_data[1], cfg.max_char)
    if answer == "no" or not os.path.isfile(cfg.file_seq_aux_task_train_data[2] + ".npz"):
        convert_conll_to_numpy_array(cfg.file_conll_aux_task_train_data[2], vocab_word2id, vocab_tag2id, vocab_char2id,
                                 cfg.file_seq_aux_task_train_data[2], cfg.max_char)

[vocab_size, dim] = np.shape(twe)
tag_size = max(vocab_id2tag, key=int) + 1
char_size = max(vocab_id2char, key=int) + 1

logger.info("loading data again.")
[main_task_all_words, main_task_all_tags, main_task_all_chars] = load_sequence_data(cfg.file_seq_main_task_train_data)
if type == "mtl2":
    [aux_task1_train_words, aux_task1_train_tags, aux_task1_train_chars] = load_sequence_data(cfg.file_seq_aux_task_train_data[0])
elif type == "mtl3":
    [aux_task1_train_words, aux_task1_train_tags, aux_task1_train_chars] = load_sequence_data(cfg.file_seq_aux_task_train_data[0])
    [aux_task2_train_words, aux_task2_train_tags, aux_task2_train_chars] = load_sequence_data(cfg.file_seq_aux_task_train_data[1])
elif type == "mtl4":
    [aux_task1_train_words, aux_task1_train_tags, aux_task1_train_chars] = load_sequence_data(cfg.file_seq_aux_task_train_data[0])
    [aux_task2_train_words, aux_task2_train_tags, aux_task2_train_chars] = load_sequence_data(cfg.file_seq_aux_task_train_data[1])
    [aux_task3_train_words, aux_task3_train_tags, aux_task3_train_chars] = load_sequence_data(cfg.file_seq_aux_task_train_data[2])

num_data = len(main_task_all_words)
num_train = int(0.95*num_data)

main_task_train_words = main_task_all_words[:num_train]
main_task_train_tags = main_task_all_tags[:num_train]
main_task_train_chars = main_task_all_chars[:num_train]
main_task_val_words = main_task_all_words[num_train:]
main_task_val_tags = main_task_all_tags[num_train:]
main_task_val_chars = main_task_all_chars[num_train:]

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
epoch_number = 0
if answer != "no":
    try:
        file_name = mod.restore_graph()
        splitted_file_name = file_name.split("-")
        epoch_number = int(splitted_file_name[-1])+1
    except Exception as e:
        logger.info("no valid checkpoint found. start to train from scratch...")


if type == "stl":
    mod.train_graph(train_word_seq=main_task_train_words, train_tag_seq=main_task_train_tags, train_char_seq=main_task_train_chars,
                    val_word_seq=main_task_val_words, val_tag_seq=main_task_val_tags, val_char_seq=main_task_val_chars
                    , word_embedding=twe, epoch_start=epoch_number, epoch_end=100, batch_size=cfg.batch_size)
elif type == "mtl2":
    mod.train_graph(main_task_train_word_seq=main_task_train_words, main_task_train_tag_seq=main_task_train_tags
                    , main_task_train_char_seq=main_task_train_chars
                    , aux_task1_train_word_seq=aux_task1_train_words, aux_task1_train_tag_seq=aux_task1_train_tags, aux_task1_train_char_seq=aux_task1_train_chars
                    , val_word_seq=main_task_val_words, val_tag_seq=main_task_val_tags, val_char_seq=main_task_val_chars
                    , word_embedding=twe, epoch_start=epoch_number, epoch_end=100, batch_size=cfg.batch_size)
elif type == "mtl3":
    mod.train_graph(main_task_train_word_seq=main_task_train_words, main_task_train_tag_seq=main_task_train_tags
                    , main_task_train_char_seq=main_task_train_chars
                    , aux_task1_train_word_seq=aux_task1_train_words, aux_task1_train_tag_seq=aux_task1_train_tags, aux_task1_train_char_seq=aux_task1_train_chars
                    , aux_task2_train_word_seq=aux_task2_train_words, aux_task2_train_tag_seq=aux_task2_train_tags, aux_task2_train_char_seq=aux_task2_train_chars
                    , val_word_seq=main_task_val_words, val_tag_seq=main_task_val_tags, val_char_seq=main_task_val_chars
                    , word_embedding=twe, epoch_start=epoch_number, epoch_end=100, batch_size=cfg.batch_size)
elif type == "mtl4":
    mod.train_graph(main_task_train_word_seq=main_task_train_words, main_task_train_tag_seq=main_task_train_tags
                    , main_task_train_char_seq=main_task_train_chars
                    , aux_task1_train_word_seq=aux_task1_train_words, aux_task1_train_tag_seq=aux_task1_train_tags, aux_task1_train_char_seq=aux_task1_train_chars
                    , aux_task2_train_word_seq=aux_task2_train_words, aux_task2_train_tag_seq=aux_task2_train_tags, aux_task2_train_char_seq=aux_task2_train_chars
                    , aux_task3_train_word_seq=aux_task3_train_words, aux_task3_train_tag_seq=aux_task3_train_tags, aux_task3_train_char_seq=aux_task3_train_chars
                    , val_word_seq=main_task_val_words, val_tag_seq=main_task_val_tags, val_char_seq=main_task_val_chars
                    , word_embedding=twe, epoch_start=epoch_number, epoch_end=100, batch_size=cfg.batch_size)
