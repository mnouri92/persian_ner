class MTLConfig():

    def __init__(self, directory_data_param, file_conll_task1_directory_param, file_conll_task2_directory_param, file_conll_task3_directory_param, file_conll_task4_directory_param, task_param):

        self.task = task_param
        self.gan = False
        self.directory_data = directory_data_param

        #task1:
        self.file_conll_task1_directory = file_conll_task1_directory_param
        self.file_conll_task1_train_data = self.file_conll_task1_directory + "train.data"
        self.file_conll_task1_validation_data = self.file_conll_task1_directory + "validation.data"
        self.file_conll_task1_test_data = self.file_conll_task1_directory + "test.data"
        self.file_seq_task1_train_data = self.file_conll_task1_train_data + ".seq"
        self.file_seq_task1_validation_data = self.file_conll_task1_validation_data + ".seq"
        self.file_seq_task1_test_data = self.file_conll_task1_test_data + ".seq"

        #task2:
        self.file_conll_task2_directory = file_conll_task2_directory_param
        self.file_conll_task2_train_data = self.file_conll_task2_directory + "train.data"
        self.file_conll_task2_validation_data = self.file_conll_task2_directory + "validation.data"
        self.file_conll_task2_test_data = self.file_conll_task2_directory + "test.data"
        self.file_seq_task2_train_data = self.file_conll_task2_train_data + ".seq"
        self.file_seq_task2_validation_data = self.file_conll_task2_validation_data + ".seq"
        self.file_seq_task2_test_data = self.file_conll_task2_test_data + ".seq"

        #task3:
        self.file_conll_task3_directory = file_conll_task3_directory_param
        self.file_conll_task3_train_data = self.file_conll_task3_directory + "train.data"
        self.file_conll_task3_validation_data = self.file_conll_task3_directory + "validation.data"
        self.file_conll_task3_test_data = self.file_conll_task3_directory + "test.data"
        self.file_seq_task3_train_data = self.file_conll_task3_train_data + ".seq"
        self.file_seq_task3_validation_data = self.file_conll_task3_validation_data + ".seq"
        self.file_seq_task3_test_data = self.file_conll_task3_test_data + ".seq"

        #task4:
        self.file_conll_task4_directory = file_conll_task4_directory_param
        self.file_conll_task4_train_data = self.file_conll_task4_directory + "train.data"
        self.file_conll_task4_validation_data = self.file_conll_task4_directory + "validation.data"
        self.file_conll_task4_test_data = self.file_conll_task4_directory + "test.data"
        self.file_seq_task4_train_data = self.file_conll_task4_train_data + ".seq"
        self.file_seq_task4_validation_data = self.file_conll_task4_validation_data + ".seq"
        self.file_seq_task4_test_data = self.file_conll_task4_test_data + ".seq"

        self.file_full_word_embedding = self.directory_data + "wiki_news_fasttext_sg_d300_w10.vec"
        self.file_trimmed_word_embedding = self.directory_data + "word_embedding.trimmed"
        self.file_word_vocab = self.directory_data + "vocab.words"
        self.file_char_vocab = self.directory_data + "vocab.chars"
        self.file_tag_vocab = self.directory_data + "vocab.tags"

        self.dir_tensoboard_log = "log/"
        self.dir_checkpoints = "chkpnts/"

        self.word_embedding_dimension        = 300
        self.char_embedding_dimension        = 100

        #LSTM_MODE
        self.lstm_model_batch_size           = 16
        self.lstm_model_hidden_size          = 300
        self.lstm_model_rnn_dropout          = 0.5
        self.lstm_model_rnn_lr               = 0.001
        self.lstm_model_max_epoch            = 100
        self.lstm_model_hidden_size_char     = 100


        #Model
        self.max_char                        = 50