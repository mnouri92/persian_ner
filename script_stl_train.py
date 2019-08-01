from stl_model import STLCharCNNWordBilstmModel
from utility import *
from stl_config import Config
logger = setup_custom_logger(__name__)

base_directory_path = sys.argv[1]

cfg = Config(base_directory_path)

answer = input("continue from previous learned models (probabely to improve them)?(yes/no)")
if answer == "no":
    logger.info("bulding vocabulray")
    [word_vocab, tag_vocab, char_vocab, max_char] = build_vocab([cfg.file_conll_train_data])

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
convert_conll_to_numpy_array(cfg.file_conll_train_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             cfg.file_seq_train_data, cfg.max_char)

[vocab_size, dim] = np.shape(twe)
tag_size = max(vocab_id2tag, key=int) + 1
char_size = max(vocab_id2char, key=int) + 1

logger.info("loading data again.")
[words, tags, chars] = load_sequence_data(cfg.file_seq_train_data)

num_data = len(words)
num_train = int(0.95*num_data)

train_words = words[:num_train]
train_tags = tags[:num_train]
train_chars = chars[:num_train]
val_words = words[num_train:]
val_tags = tags[num_train:]
val_chars = chars[num_train:]


mod = STLCharCNNWordBilstmModel(vocab_size, dim, tag_size, cfg.max_char, cfg.char_embedding_dimension, cfg.lstm_model_hidden_size
                                , cfg.lstm_model_rnn_lr, cfg.dir_tensoboard_log, cfg.dir_checkpoints, char_size)
mod.build_graph()
epoch_number = 0
if answer != "no":
    file_name = mod.restore_graph()
    splitted_file_name = file_name.split("-")
    epoch_number = int(splitted_file_name[-1])+1

mod.train_graph(train_word_seq=train_words, train_tag_seq=train_tags, train_char_seq=train_chars
                , word_embedding=twe, epoch_start=epoch_number, epoch_end = 100, batch_size = cfg.lstm_model_batch_size)