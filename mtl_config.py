class MTLConfig():

    def __init__(self, base_directory_path):

        self.base_directory_path = base_directory_path
        self.base_directory_data = self.base_directory_path + "data/"
        self.base_directory_model = self.base_directory_path + "model/"

        #NER_Bijankhan:
        self.file_conll_task1_directory = self.base_directory_data + "ner_bijankhan/"
        self.file_conll_task1_train_data = self.file_conll_task1_directory + "train.data"
        self.file_conll_task1_validation_data = self.file_conll_task1_directory + "validation.data"
        self.file_seq_task1_train_data = self.file_conll_task1_train_data + ".seq"
        self.file_seq_task1_validation_data = self.file_conll_task1_validation_data + ".seq"

        #NER_ArmanPerso:
        self.file_conll_task2_directory = self.base_directory_data + "ner_armanperso/"
        self.file_conll_task2_train_data = self.file_conll_task2_directory + "train.data"
        self.file_seq_task2_train_data = self.file_conll_task2_train_data + ".seq"

        self.file_full_word_embedding = self.base_directory_path + "wiki_news_fasttext_sg_d300_w10.vec"
        self.file_trimmed_word_embedding = self.base_directory_model + "word_embedding.trimmed"
        self.file_word_vocab = self.base_directory_model + "vocab.words"
        self.file_char_vocab = self.base_directory_model + "vocab.chars"
        self.file_tag_vocab = self.base_directory_model + "vocab.tags"

        self.dir_tensoboard_log = "log/"
        self.dir_checkpoints = self.base_directory_model

        self.word_embedding_dimension        = 300
        self.char_embedding_dimension        = 100

        #LSTM_MODE
        self.lstm_model_batch_size           = 16
        self.lstm_model_hidden_size          = 300
        self.lstm_model_rnn_dropout          = 0.5
        self.lstm_model_rnn_lr               = 0.001
        self.lstm_model_max_epoch            = 30
        self.lstm_model_hidden_size_char     = 100


        #Model
        self.max_char                        = 65