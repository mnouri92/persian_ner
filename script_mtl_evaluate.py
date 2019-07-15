from model import MTL2CharCNNWordBilstmModel

from charnomalizer import CharNormalizer
from hazm import sent_tokenize, word_tokenize

from config import MTLConfig
from utility import *
import os.path
import pickle

from os import listdir
from os.path import isfile, join

test_conll_data_directory_input = sys.argv[1]
test_conll_data_directory_output = sys.argv[2]

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
mod = MTL2CharCNNWordBilstmModel(vocab_size, dim, tag_size, tag_size, cfg, char_size)
mod.build_graph()
mod.restore_graph()
vocab_id2word[-1] = "OOV"
onlyfiles = [f for f in listdir(test_conll_data_directory_input) if isfile(join(test_conll_data_directory_input, f))]
cou=1
for filename in onlyfiles:
    print("==================={}/{}:{}=======================".format(cou, len(onlyfiles), filename))
    cou+=1
    filename_without_extension = os.path.splitext(filename)[0]
    output_file = join(test_conll_data_directory_output, filename_without_extension +".predict")
    input_file = join(test_conll_data_directory_input, filename)
    text = ""
    with open(input_file, 'r') as f:
        text = f.read()

    lines = text.split('\n')
    words = []
    other_info = []
    current_sentence_word = []
    current_sentence_other_info = []
    counter = 0
    for line in lines:
        if counter % 1000 == 0:
            print("{}/{}".format(str(counter), len(lines)))
        counter += 1

        if len(line.strip()) == 0:
            if len(current_sentence_word) > 0:
                words.append(current_sentence_word)
                other_info.append(current_sentence_other_info)
            current_sentence_word = []
            current_sentence_other_info = []
        else:
            current_sentence_word.append(normalizer.normalize(line.split()[0].strip()))
            current_sentence_other_info += line.split()[2:]

    word_ids = []
    char_ids = []
    counter = 0
    for sentence in words:
        if counter % 1000 == 0:
            print("{}/{}".format(str(counter), len(words)))
        counter += 1
        current_sentence_word_id = []
        current_sentence_char_id = []
        for word in sentence:
            current_id, twe = update_word_vocab(word, vocab_id2word, vocab_word2id, twe, word_embedding, avg_vector)
            current_sentence_word_id.append(current_id)

            current_word_chars = [vocab_char2id[x] for x in word if x in vocab_char2id.keys()]
            current_word_chars += [0] * (cfg.max_char - len(current_word_chars))

            current_sentence_char_id.append(current_word_chars)

        word_ids.append(current_sentence_word_id)
        char_ids.append(current_sentence_char_id)

    convert_conll_to_numpy_array(input_file, vocab_word2id, vocab_tag2id, vocab_char2id,
                                 input_file + ".seq", cfg.max_char)

    [test_words, test_tags, test_chars] = load_sequence_data(input_file + ".seq")
    mod.evaluate_model(test_word_seq=test_words, test_tag_seq=test_tags, test_char_seq=test_chars, word_embedding=twe
                       , id2word=vocab_id2word, id2tag=vocab_id2tag, result_file_path=output_file, task_number=1)



