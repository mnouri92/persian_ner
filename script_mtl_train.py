from model.mtl2_model import MTL2Model
from model.mtl3_model import MTL3Model
from model.mtl4_model import MTL4Model
from model.mtl2_gan_model import MTL2GANModel
from model.mtl3_gan_model import MTL3GANModel
from model.mtl2_char_cnn_word_bilstm_model import MTL2CharCNNWordBilstmModel
from model.mtl3_char_cnn_word_bilstm_model import MTL3CharCNNWordBilstmModel
from model.mtl4_char_cnn_word_bilstm_model import MTL4CharCNNWordBilstmModel


from common.utility import *
from common.mtl_config import MTLConfig

numtask = sys.argv[1]

if numtask == "mtl2":
    task1 = sys.argv[2]
    task2 = sys.argv[3]
    task3 = ""
    task4 = ""
elif numtask == "mtl3":
    task1 = sys.argv[2]
    task2 = sys.argv[3]
    task3 = sys.argv[4]
    task4 = ""
elif numtask == "mtl4":
    task1 = sys.argv[2]
    task2 = sys.argv[3]
    task3 = sys.argv[4]
    task4 = sys.argv[5]
elif numtask == "mtl2charcnnwordbilstm":
    task1 = sys.argv[2]
    task2 = sys.argv[3]
    task3 = ""
    task4 = ""
elif numtask == "mtl3charcnnwordbilstm":
    task1 = sys.argv[2]
    task2 = sys.argv[3]
    task3 = sys.argv[4]
    task4 = ""
elif numtask == "mtl4charcnnwordbilstm":
    task1 = sys.argv[2]
    task2 = sys.argv[3]
    task3 = sys.argv[4]
    task4 = sys.argv[5]

cfg = MTLConfig(directory_data_param="/home/itrc/hadi/workspace/persian_ner/files/mtl/data/"
                             , file_conll_task1_directory_param="/home/itrc/hadi/workspace/persian_ner/files/mtl/data/" + task1 + "/"
                             , file_conll_task2_directory_param="/home/itrc/hadi/workspace/persian_ner/files/mtl/data/" + task2 + "/"
                             , file_conll_task3_directory_param="/home/itrc/hadi/workspace/persian_ner/files/mtl/data/" + task3 + "/"
                             , file_conll_task4_directory_param="/home/itrc/hadi/workspace/persian_ner/files/mtl/data/" + task4 + "/"
                             , task_param=numtask)

[vocab_id2tag, vocab_tag2id] = load_vocab(cfg.file_tag_vocab)
[vocab_id2word, vocab_word2id] = load_vocab(cfg.file_word_vocab)
[vocab_id2char, vocab_char2id] = load_vocab(cfg.file_char_vocab)

twe = load_trimmed_word_embeddings(cfg.file_trimmed_word_embedding)
[vocab_size, dim] = np.shape(twe)
tag_size = max(vocab_id2tag, key=int) + 1
char_size = max(vocab_id2char, key=int) + 1

[validation_words, validation_tags, validation_chars] = load_sequence_data(cfg.file_seq_task1_validation_data)

[task1_train_words, task1_train_tags, task1_train_chars] = load_sequence_data(cfg.file_seq_task1_train_data)

[task2_train_words, task2_train_tags, task2_train_chars] = load_sequence_data(cfg.file_seq_task2_train_data)

if cfg.task == "mtl3" or cfg.task == "mtl3charcnnwordbilstm":
    [task3_train_words, task3_train_tags, task3_train_chars] = load_sequence_data(cfg.file_seq_task3_train_data)

if cfg.task == "mtl4" or cfg.task == "mtl4charcnnwordbilstm":
    [task3_train_words, task3_train_tags, task3_train_chars] = load_sequence_data(cfg.file_seq_task3_train_data)
    [task4_train_words, task4_train_tags, task4_train_chars] = load_sequence_data(cfg.file_seq_task4_train_data)


if cfg.task == "mtl2":
    if cfg.gan:
        mod = MTL2GANModel(vocab_size, dim, tag_size, tag_size)
        mod.build_graph()
        mod.train_graph(task1_train_word_seq=task1_train_words, task1_train_tag_seq=task1_train_tags
                        , task2_train_word_seq=task2_train_words, task2_train_tag_seq=task2_train_tags
                        , word_embedding=twe)
    else:
        mod = MTL2CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, cfg)
        mod.build_graph()
        mod.train_graph(task1_train_word_seq=task1_train_words, task1_train_tag_seq=task1_train_tags
                        , task2_train_word_seq=task2_train_words, task2_train_tag_seq=task2_train_tags
                        , word_embedding=twe)

if cfg.task == "mtl3":
    if cfg.gan:
        mod = MTL3GANModel(vocab_size, dim, tag_size, tag_size, tag_size)
        mod.build_graph()
        mod.train_graph(task1_train_word_seq=task1_train_words, task1_train_tag_seq=task1_train_tags
                        , task2_train_word_seq=task2_train_words, task2_train_tag_seq=task2_train_tags
                        , task3_train_word_seq=task3_train_words, task3_train_tag_seq=task3_train_tags
                        , word_embedding=twe)
    else:
        mod = MTL3Model(vocab_size, dim, tag_size, tag_size, tag_size, cfg)
        mod.build_graph()
        mod.train_graph(task1_train_word_seq=task1_train_words, task1_train_tag_seq=task1_train_tags
                        , task2_train_word_seq=task2_train_words, task2_train_tag_seq=task2_train_tags
                        , task3_train_word_seq=task3_train_words, task3_train_tag_seq=task3_train_tags
                        , word_embedding=twe)

if cfg.task == "mtl4":
    if cfg.gan:
        print("nothing to do")
    else:
        mod = MTL4Model(vocab_size, dim, tag_size, tag_size, tag_size, tag_size, cfg)
        mod.build_graph()
        mod.train_graph(task1_train_word_seq=task1_train_words, task1_train_tag_seq=task1_train_tags
                        , task2_train_word_seq=task2_train_words, task2_train_tag_seq=task2_train_tags
                        , task3_train_word_seq=task3_train_words, task3_train_tag_seq=task3_train_tags
                        , task4_train_word_seq=task4_train_words, task4_train_tag_seq=task4_train_tags
                        , word_embedding=twe)

if cfg.task == "mtl2charcnnwordbilstm":
    mod = MTL2CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, cfg, char_size)
    mod.build_graph()
    mod.train_graph(task1_train_word_seq=task1_train_words, task1_train_tag_seq=task1_train_tags, task1_train_char_seq=task1_train_chars
                    , task2_train_word_seq=task2_train_words, task2_train_tag_seq=task2_train_tags, task2_train_char_seq=task2_train_chars
                    , word_embedding=twe)

if cfg.task == "mtl3charcnnwordbilstm":
    mod = MTL3CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, tag_size, cfg)
    mod.build_graph()
    mod.train_graph(task1_train_word_seq=task1_train_words, task1_train_tag_seq=task1_train_tags, task1_train_char_seq=task1_train_chars
                    , task2_train_word_seq=task2_train_words, task2_train_tag_seq=task2_train_tags, task2_train_char_seq=task2_train_chars
                    , task3_train_word_seq=task3_train_words, task3_train_tag_seq=task3_train_tags, task3_train_char_seq=task3_train_chars
                    , word_embedding=twe)

if cfg.task == "mtl4charcnnwordbilstm":
    mod = MTL4CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, tag_size, tag_size, cfg)
    mod.build_graph()
    mod.train_graph(task1_train_word_seq=task1_train_words, task1_train_tag_seq=task1_train_tags, task1_train_char_seq=task1_train_chars
                    , task2_train_word_seq=task2_train_words, task2_train_tag_seq=task2_train_tags, task2_train_char_seq=task2_train_chars
                    , task3_train_word_seq=task3_train_words, task3_train_tag_seq=task3_train_tags, task3_train_char_seq=task3_train_chars
                    , task4_train_word_seq=task4_train_words, task4_train_tag_seq=task4_train_tags, task4_train_char_seq=task4_train_chars
                    , word_embedding=twe)





