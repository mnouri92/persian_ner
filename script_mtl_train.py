from mtl_model import MTL2CharCNNWordBilstmModel
from utility import *
from config import MTLConfig

logger = setup_custom_logger(__name__)

cfg = MTLConfig("files/")

answer = input("continue from previous learned models (probabely to improve them)?(yes/no)")
if answer == "no":
    logger.info("bulding vocabulray")
    [word_vocab, tag_vocab, char_vocab, max_char] = build_vocab([
        cfg.file_conll_task1_train_data, cfg.file_conll_task1_validation_data, cfg.file_conll_task2_train_data])

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
convert_conll_to_numpy_array(cfg.file_conll_task1_train_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             cfg.file_seq_task1_train_data, cfg.max_char)
convert_conll_to_numpy_array(cfg.file_conll_task1_validation_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             cfg.file_seq_task1_validation_data, cfg.max_char)
convert_conll_to_numpy_array(cfg.file_conll_task2_train_data, vocab_word2id, vocab_tag2id, vocab_char2id,
                             cfg.file_seq_task2_train_data, cfg.max_char)

[vocab_size, dim] = np.shape(twe)
tag_size = max(vocab_id2tag, key=int) + 1
char_size = max(vocab_id2char, key=int) + 1

logger.info("loading data again.")
[validation_words, validation_tags, validation_chars] = load_sequence_data(cfg.file_seq_task1_validation_data)
[task1_train_words, task1_train_tags, task1_train_chars] = load_sequence_data(cfg.file_seq_task1_train_data)
[task2_train_words, task2_train_tags, task2_train_chars] = load_sequence_data(cfg.file_seq_task2_train_data)

mod = MTL2CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, cfg, char_size)
mod.build_graph()
epoch_number = 0
if answer != "no":
    file_name = mod.restore_graph()
    splitted_file_name = file_name.split("-")
    epoch_number = int(splitted_file_name[-1])+1


mod.train_graph(task1_train_word_seq=task1_train_words, task1_train_tag_seq=task1_train_tags, task1_train_char_seq=task1_train_chars
                , task2_train_word_seq=task2_train_words, task2_train_tag_seq=task2_train_tags, task2_train_char_seq=task2_train_chars
                , word_embedding=twe, epoch_start=epoch_number)