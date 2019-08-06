class Config():

    def __init__(self, main_task_directory_path, aux_task_directory_path, model_directory_path, file_full_word_embedding):

        self.main_task_directory_path               = main_task_directory_path + "/"
        self.main_task_data_directory_path          = main_task_directory_path + "data/"

        self.file_conll_main_task_train_data        = self.main_task_data_directory_path + "train.data"
        self.file_seq_main_task_train_data          = self.file_conll_main_task_train_data + ".seq"

        self.aux_task_data_directory_path       = [path + "/data/" for path in aux_task_directory_path]

        self.file_conll_aux_task_train_data     = [path + "train.data" for path in self.aux_task_data_directory_path]
        self.file_seq_aux_task_train_data       = [path + ".seq" for path in self.file_conll_aux_task_train_data]

        self.file_full_word_embedding               = file_full_word_embedding

        self.dir_tensoboard_log                     = "log/"

        self.main_task_model_directory_path         = model_directory_path + "/"

        self.dir_checkpoints                        = self.main_task_model_directory_path
        self.file_trimmed_word_embedding            = self.main_task_model_directory_path + "word_embedding.trimmed"
        self.file_word_vocab                        = self.main_task_model_directory_path + "vocab.words"
        self.file_char_vocab                        = self.main_task_model_directory_path + "vocab.chars"
        self.file_tag_vocab                         = self.main_task_model_directory_path + "vocab.tags"
        self.word_embedding_dimension               = 300
        self.char_embedding_dimension               = 100
        self.batch_size                             = 16
        self.wrd_lstm_hidden_size                   = 300
        self.dropout                                = 0.5
        self.learning_rate                          = 0.001
        self.max_epoch                              = 40
        self.char_lstm_hidden_size                  = 100
        self.max_char                               = 65