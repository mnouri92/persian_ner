import numpy as np
import logging
import sys

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('logs.txt', mode='a')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger(__name__)


def build_vocab(files):
    '''
    Constructs vocabulary file from a Conll file

    :param file_name:   a file in conll format. the first column in each line is the word. a line may be empty to show the sentence boundary

    :return: vocabulary of the file
    '''
    word_vocab = set()
    tag_vocab = set()
    char_vocab = set()
    max_char = 0;
    for file_name in files:
        logger.info("start to create the vocab from train file: {}".format(file_name))
        with open(file_name) as f:
            for line in f:
                words = line.split()
                if len(words) <= 1:
                    continue;
                word = words[0]
                tag = words[1]
                word_vocab.add(word)
                tag_vocab.add(tag)
                chars = list(word)
                max_char = max(len(chars),max_char)
                char_vocab.update(chars)
        logger.info("done. number of unique words: {} number of unique tags: {} number of unique characters: {}".format(len(word_vocab), len(tag_vocab), len(char_vocab)))
    return sorted(word_vocab), sorted(tag_vocab), sorted(char_vocab), max_char

def dump_vocab(vocab, file_vocab_name):
    """
    write all words in the vocab in a text file line by line
    :param vocab: vocabulary set
    :param file_vocab_name: output file name
    :return:
    """
    logger.info("writing vocabulary in the file: {}".format(file_vocab_name))
    with open(file_vocab_name, 'w') as f:
        for word in vocab:
            f.write(word + "\n")
    logger.info("done.")

def load_vocab(file_vocab_name):
    """
    load the vocabulary file

    :param file_vocab_name:  output file name
    :return:
    """
    logger.info("start to load vocabularies from the dumped file: ".format(file_vocab_name))
    vocab_id2word = dict();
    vocab_word2id = dict();
    with open(file_vocab_name) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            vocab_id2word[idx] = word
            vocab_word2id[word] = idx
    logger.info("done. number of read words: {}".format(len(vocab_word2id)))
    return vocab_id2word, vocab_word2id

def update_word_vocab(word, vocab_id2word, vocab_word2id, trimmed_word_embedding=[], full_word_embedding={}, avg_embedding_vector=[]):
    final_id = len(vocab_word2id)
    if word not in vocab_word2id.keys():
        assert np.shape(trimmed_word_embedding)[0] == final_id
        # word_embedding = np.zeros_like(avg_embedding_vector)

        if word in full_word_embedding.keys():
            word_embedding = [full_word_embedding[word]]
        else:
            word_embedding = avg_embedding_vector

        vocab_word2id[word] = final_id
        vocab_id2word[final_id] = word
        final_id += 1
        trimmed_word_embedding = np.append(trimmed_word_embedding, word_embedding, axis = 0)
        assert np.shape(trimmed_word_embedding)[0] == final_id
    return vocab_word2id[word], trimmed_word_embedding

def update_char_vocab(char, vocab_id2char, vocab_char2id):
    if char in vocab_char2id.keys():
        return vocab_char2id[char]
    else:
        return 0


def update_vocab(word, vocab_id2word, vocab_word2id):
    final_id = len(vocab_word2id)
    if word not in vocab_word2id.keys():
        vocab_word2id[word] = final_id
        vocab_id2word[final_id] = word
        final_id += 1
    return vocab_word2id[word]


def dump_trimmed_word_embeddings(vocab, we_file_name, trimmed_file_name, dim):
    """Saves word embedding vectors in numpy array

        Args:
            vocab: dictionary vocab[word] = index
            we_file_name: a path to a word embedding file
            trimmed_file_name: a path where to store a matrix in npy
            dim: (int) dimension of embeddings
    """
    logger.info("start to trim the word embedding vectors: {}".format(we_file_name))
    embeddings = np.zeros([len(vocab), dim])
    with open(we_file_name) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    logger.info("number of words: {}".format(len(embeddings)))
    np.savez_compressed(trimmed_file_name, embeddings=embeddings)
    logger.info("trimmed file is stored in: {}".format(trimmed_file_name))

def get_full_word_embeddings(we_file_name, dim):
    logger.info("start to read the word embedding: {}".format(we_file_name))
    embeddings = {}
    avg = np.zeros(shape=[1,dim])
    with open(we_file_name) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = np.array([float(x) for x in line[1:]])
            avg += embedding
            embeddings[word] = embedding

    word_num = len(embeddings)
    avg /= word_num

    logger.info("number of words: {}".format(word_num))
    return embeddings, avg

def load_trimmed_word_embeddings(trimmed_file_name):
    """
    load the previously saved trimmed word embeddings
    :param trimmed_file_name: the path
    :return: the word embedding
    """
    trimmed_file_name += ".npz"
    logger.info("start to load the trimmed file: {}".format(trimmed_file_name))
    with np.load(trimmed_file_name) as data:
        we = data["embeddings"]
        [w,d] = np.shape(we)
        logger.info("data is loaded. vocab size: {}  dim: {}".format(w,d))
        return we

def convert_conll_to_numpy_array(file_name, vocab_word, vocab_tag, vocab_char, output_file, max_char, OOV_index=-1):
    logger.info("start to convert the conll file to word/tag sequence: {}".format(file_name))

    all_sentences_word = []
    all_sentences_tag = []
    all_sentences_char = []
    with open(file_name, 'r') as f:
        current_sentence_word = []
        current_sentence_tag = []
        current_sentence_char = []
        for line in f:
            if len(line.strip()) == 0:
                if(len(current_sentence_word) > 0 and len(current_sentence_word) < 70000):
                    assert len(current_sentence_word) == len(current_sentence_tag)
                    all_sentences_word.append(current_sentence_word)
                    all_sentences_tag.append(current_sentence_tag)
                    all_sentences_char.append(current_sentence_char)
                current_sentence_word = []
                current_sentence_tag = []
                current_sentence_char = []
            else:
                splitted_line = line.split()
                assert len(splitted_line) == 2
                word = splitted_line[0]
                tag = splitted_line[1]
                if word in vocab_word.keys():
                    current_sentence_word.append(vocab_word[word])
                else:
                    current_sentence_word.append(OOV_index)
                current_sentence_tag.append(vocab_tag[tag])
                current_word_chars = [vocab_char[x] for x in word]
                current_word_chars += [0]*(max_char-len(current_word_chars))
                current_sentence_char.append(current_word_chars)

    np.savez_compressed(output_file, words=all_sentences_word, tags=all_sentences_tag, chars = all_sentences_char)
    logger.info("sequences are stored successfully: {}".format(output_file))
    return

def load_sequence_data(sequence_data_file_name):
    """
    load the previously saved sequence data
    :param sequence_data_file_name
    :return: word, tag
    """
    sequence_data_file_name += ".npz"
    logger.info("start to load the sequence file: {}".format(sequence_data_file_name))
    with np.load(sequence_data_file_name) as data:
        words = data["words"]
        tags = data["tags"]
        chars = data["chars"]
        return words, tags, chars
#
# def numpy_fillna(data):
#     # Get lengths of each row of data
#     lens = np.array([len(i) for i in data])
#
#     # Mask of valid places in each row
#     mask = np.arange(lens.max()) < lens[:,None]
#
#     # Setup output array and put elements from data into masked positions
#     out = np.zeros(mask.shape, dtype=data.dtype)
#     out[mask] = np.concatenate(data)
#     return out

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length

def remove_padding(data, seq_len):
    num_seq = len(seq_len)
    nonpadded_data = []
    for i in range(num_seq):
        end = seq_len[i]
        nonpadded_data.append(data[i][:end])
    return nonpadded_data

def make_flat(seq):
    return np.vstack(seq)

BEGIN = 'b-'
INTER = 'i-'
def extract_entities(words, labels_txt):
    all_entities = {}
    all_tags = {}
    current_entity = []
    current_tag = ''
    for i in range(len(labels_txt)):
        for j in range(len(labels_txt[i])):
            current_tag = str(labels_txt[i][j]).lower()
            prev_tag = 'o'
            if(j >= 1):
                prev_tag = str(labels_txt[i][j-1]).lower()
            if prev_tag == 'o':
                assert len(current_entity) == 0
                if current_tag.startswith(BEGIN):
                    current_entity.append(words[i][j])
            elif prev_tag.startswith(BEGIN):
                assert len(current_entity) == 1
                if current_tag.startswith(BEGIN):
                    final_entity = ' '.join(current_entity)
                    if final_entity not in all_entities.keys():
                        all_entities[final_entity] = 0
                        all_tags[final_entity] = prev_tag[2:]
                    all_entities[final_entity] = all_entities[final_entity] + 1
                    current_entity = []
                    current_entity.append(words[i][j])
                elif current_tag.startswith(INTER):
                    if prev_tag[2:] == current_tag[2:]:
                        current_entity.append(words[i][j])
                    else:
                        final_entity = ' '.join(current_entity)
                        if final_entity not in all_entities.keys():
                            all_entities[final_entity] = 0
                            all_tags[final_entity] = prev_tag[2:]
                        all_entities[final_entity] = all_entities[final_entity] + 1
                        current_entity = []
                else:
                    final_entity = ' '.join(current_entity)
                    if final_entity not in all_entities.keys():
                        all_entities[final_entity] = 0
                        all_tags[final_entity] = prev_tag[2:]
                    all_entities[final_entity] = all_entities[final_entity] + 1
                    current_entity = []
            elif prev_tag.startswith(INTER):
                if len(current_entity) == 0:
                    if current_tag.startswith(BEGIN):
                        current_entity.append(words[i][j])
                    continue
                if current_tag.startswith(BEGIN):
                    final_entity = ' '.join(current_entity)
                    if final_entity not in all_entities.keys():
                        all_entities[final_entity] = 0
                        all_tags[final_entity] = prev_tag[2:]
                    all_entities[final_entity] = all_entities[final_entity] + 1
                    current_entity = []
                    current_entity.append(words[i][j])
                elif current_tag.startswith(INTER):
                    if prev_tag[2:] == current_tag[2:]:
                        current_entity.append(words[i][j])
                    else:
                        final_entity = ' '.join(current_entity)
                        if final_entity not in all_entities.keys():
                            all_entities[final_entity] = 0
                            all_tags[final_entity] = prev_tag[2:]
                        all_entities[final_entity] = all_entities[final_entity] + 1
                        current_entity = []
                else:
                    final_entity = ' '.join(current_entity)
                    if final_entity not in all_entities.keys():
                        all_entities[final_entity] = 0
                        all_tags[final_entity] = prev_tag[2:]
                    all_entities[final_entity] = all_entities[final_entity] + 1
                    current_entity = []
        if len(current_entity) > 0:
            final_entity = ' '.join(current_entity)
            if final_entity not in all_entities.keys():
                all_entities[final_entity] = 0
                all_tags[final_entity] = current_tag[2:]
            all_entities[final_entity] = all_entities[final_entity] + 1
            current_entity = []

    return all_entities, all_tags





