from model.mtl_model import MTL2CharCNNWordBilstmModel
from model.stl_model import STLCharCNNWordBilstmModel
from common.utility import *
from common.mtl_config import MTLConfig
from common.config import Config


logger = setup_custom_logger(__name__)

type = sys.argv[1]
model_path = sys.argv[2]
main_task_directory_path = sys.argv[3]
if type == "mtl":
    aux_task_directory_path = sys.argv[4]

if type == "stl":
    cfg = Config(main_task_directory_path, model_path)
    all_files = [cfg.file_conll_main_task_train_data]
elif type == "mtl":
    cfg = MTLConfig(main_task_directory_path, aux_task_directory_path, model_path)
    all_files = [cfg.file_conll_main_task_train_data, cfg.file_conll_aux_task_train_data]

answer = input("continue from previous learned models (probabely to improve them)?(yes/no)")
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
if type == "mtl":
    convert_conll_to_numpy_array(cfg.file_conll_aux_task_train_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 cfg.file_seq_aux_task_train_data, cfg.max_char)

[vocab_size, dim] = np.shape(twe)
tag_size = max(vocab_id2tag, key=int) + 1
char_size = max(vocab_id2char, key=int) + 1

logger.info("loading data again.")
[main_task_all_words, main_task_all_tags, main_task_all_chars] = load_sequence_data(cfg.file_seq_main_task_train_data)
if type == "mtl":
    [aux_task_train_words, aux_task_train_tags, aux_task_train_chars] = load_sequence_data(cfg.file_seq_aux_task_train_data)

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
elif type == "mtl":
    mod = MTL2CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, cfg.max_char, cfg.char_embedding_dimension,
                                     cfg.wrd_lstm_hidden_size
                                     , cfg.learning_rate, cfg.dir_tensoboard_log, cfg.dir_checkpoints, char_size)

mod.build_graph()
epoch_number = 0
if answer != "no":
    file_name = mod.restore_graph()
    splitted_file_name = file_name.split("-")
    epoch_number = int(splitted_file_name[-1])+1


if type == "stl":
    mod.train_graph(train_word_seq=main_task_train_words, train_tag_seq=main_task_train_tags, train_char_seq=main_task_train_chars,
                    val_word_seq=main_task_val_words, val_tag_seq=main_task_val_tags, val_char_seq=main_task_val_chars
                    , word_embedding=twe, epoch_start=epoch_number, epoch_end=100, batch_size=cfg.batch_size)
elif type == "mtl":
    mod.train_graph(main_task_train_word_seq=main_task_train_words, main_task_train_tag_seq=main_task_train_tags,
                    main_task_train_char_seq=main_task_train_chars
                    , aux_task_train_word_seq=aux_task_train_words, aux_task_train_tag_seq=aux_task_train_tags,
                    aux_task_train_char_seq=aux_task_train_chars
                    , val_word_seq=main_task_val_words, val_tag_seq=main_task_val_tags, val_char_seq=main_task_val_chars
                    , word_embedding=twe, epoch_start=epoch_number, epoch_end=100, batch_size=cfg.batch_size)
