from mtl2_char_cnn_word_bilstm_model import MTL2CharCNNWordBilstmModel

from charnomalizer import CharNormalizer
from hazm import sent_tokenize, word_tokenize

from mtl_config import MTLConfig
from utility import *
import os.path
import pickle

test_conll_data_file_path = sys.argv[1]

cfg = MTLConfig("files/")

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

normalizer = CharNormalizer()

output_file = test_conll_data_file_path + '.out'

text = ""
with open(test_conll_data_file_path, 'r') as f:
    text = f.read()

lines = text.split('\n')
words = []
tags = []
other_info = []
current_sentence_word = []
current_sentence_tag = []
current_sentence_other_info = []
counter = 0
for line in lines:
    print("{}/{}".format(str(counter), len(lines)), end="\r")
    counter += 1

    if len(line.strip()) == 0:
        if len(current_sentence_word)>0:
            words.append(current_sentence_word)
            tags.append(current_sentence_tag)
            other_info.append(current_sentence_other_info)
        current_sentence_word = []
        current_sentence_other_info = []
    else:
        current_sentence_word.append(normalizer.normalize(line.split()[0].strip()))
        current_sentence_tag.append(line.split()[1].strip())
        current_sentence_other_info += line.split()[2:]

word_ids = []

tag_ids = []
char_ids = []
counter = 0
for sentence in words:
    print("{}/{}".format(str(counter), len(words)), end="\r")
    counter += 1
    current_sentence_word_id = []
    current_sentence_char_id = []
    for word in sentence:
        current_id, twe = update_word_vocab(word, vocab_id2word, vocab_word2id, twe, word_embedding, avg_vector)
        # current_id = update_vocab(word, vocab_id2word, vocab_word2id)
        current_sentence_word_id.append(current_id)

        current_word_chars = [vocab_char2id[x] for x in word if x in vocab_char2id.keys()]
        current_word_chars += [0] * (cfg.max_char - len(current_word_chars))

        current_sentence_char_id.append(current_word_chars)
        # if(word in word_embedding.keys()):
        #     twe[current_id] = np.asarray(word_embedding[word])
        # else:
        #     twe = np.append(twe, avg_vector, axis = 0)

    word_ids.append(current_sentence_word_id)
    char_ids.append(current_sentence_char_id)

counter = 0
for sentence in tags:
    print("{}/{}".format(str(counter), len(tags)), end="\r")
    counter += 1
    current_sentence_tag_id = []
    for tag in sentence:
        current_sentence_tag_id.append(vocab_tag2id[tag])

    tag_ids.append(current_sentence_tag_id)

word_ids = np.array(word_ids)
char_ids = np.array(char_ids)
tag_ids = np.array(tag_ids)

mod = MTL2CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, cfg)
mod.build_graph()
mod.restore_graph()
mod.evaluate_model(test_word_seq=word_ids, test_tag_seq=tag_ids, test_char_seq= char_ids, word_embedding=twe
                   , id2word=vocab_id2word, id2tag=vocab_id2tag, result_file_path='results', task_number=1)
