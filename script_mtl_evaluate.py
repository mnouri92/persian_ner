from model.mtl2_char_cnn_word_bilstm_model import MTL2CharCNNWordBilstmModel

from mtl_config import MTLConfig
from utility import *
import time

test_conll_data_file_path = sys.argv[1]

cfg = MTLConfig("files/")

[vocab_id2tag, vocab_tag2id] = load_vocab(cfg.file_tag_vocab)
[vocab_id2word, vocab_word2id] = load_vocab(cfg.file_word_vocab)
[vocab_id2char, vocab_char2id] = load_vocab(cfg.file_char_vocab)

twe = load_trimmed_word_embeddings(cfg.file_trimmed_word_embedding)
[vocab_size, dim] = np.shape(twe)

tag_size = max(vocab_id2tag, key=int) + 1
char_size = max(vocab_id2char, key=int) + 1

[test_words, test_tags, test_chars] = load_sequence_data(cfg.file_seq_task1_test_data)
cfg.max_char = np.shape(test_chars[0])[1]

if cfg.task == "mtl2":
    if cfg.gan:
        mod = MTL2GANModel(vocab_size, dim, tag_size, tag_size)
        mod.build_graph()
        mod.restore_graph()
        a = time.time()
        mod.evaluate_model(test_word_seq=test_words, test_tag_seq=test_tags, word_embedding=twe
                           , id2word=vocab_id2word, id2tag=vocab_id2tag, result_file_path='results', task_number=1)
        print(time.time() - a)
    else:
        mod = MTL2Model(vocab_size, dim, tag_size, tag_size, cfg)
        mod.build_graph()
        mod.restore_graph()
        a = time.time()
        mod.evaluate_model(test_word_seq=test_words, test_tag_seq=test_tags, word_embedding=twe
                           , id2word=vocab_id2word, id2tag=vocab_id2tag, result_file_path='results', task_number=1)
        print(time.time() - a)
elif cfg.task == "mtl3":
    mod = MTL3Model(vocab_size, dim, tag_size, tag_size, tag_size, cfg)
    mod.build_graph()
    mod.restore_graph()
    a = time.time()
    mod.evaluate_model(test_word_seq=test_words, test_tag_seq=test_tags, word_embedding=twe
                       , id2word=vocab_id2word, id2tag=vocab_id2tag, result_file_path='results', task_number=1)
    print(time.time() - a)
elif cfg.task == "mtl4":
    mod = MTL4Model(vocab_size, dim, tag_size, tag_size, tag_size, tag_size, cfg)
    mod.build_graph()
    mod.restore_graph()
    a = time.time()
    mod.evaluate_model(test_word_seq=test_words, test_tag_seq=test_tags, word_embedding=twe
                      , id2word=vocab_id2word, id2tag=vocab_id2tag, result_file_path='results', task_number=1)
    print(time.time()-a)
elif cfg.task == "mtl2charcnnwordbilstm":
    mod = MTL2CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, cfg)
    mod.build_graph()
    mod.restore_graph()
    a = time.time()
    mod.evaluate_model(test_word_seq=test_words, test_tag_seq=test_tags, test_char_seq= test_chars, word_embedding=twe
                       , id2word=vocab_id2word, id2tag=vocab_id2tag, result_file_path='results', task_number=1)
    print(time.time() - a)
elif cfg.task == "mtl3charcnnwordbilstm":
    mod = MTL3CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, tag_size, cfg)
    mod.build_graph()
    mod.restore_graph()
    a = time.time()
    mod.evaluate_model(test_word_seq=test_words, test_tag_seq=test_tags, test_char_seq=test_chars, word_embedding=twe
                       , id2word=vocab_id2word, id2tag=vocab_id2tag, result_file_path='results', task_number=1)
    print(time.time() - a)
elif cfg.task == "mtl4charcnnwordbilstm":
    mod = MTL4CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, tag_size, tag_size, cfg)
    mod.build_graph()
    mod.restore_graph()
    a = time.time()
    mod.evaluate_model(test_word_seq=test_words, test_tag_seq=test_tags, test_char_seq= test_chars, word_embedding=twe
                       , id2word=vocab_id2word, id2tag=vocab_id2tag, result_file_path='results', task_number=1)
    print(time.time() - a)

