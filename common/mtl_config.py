from common.config import Config

class MTLConfig(Config):

    def __init__(self, main_task_directory_path, aux_task_directory_path, model_path, file_full_word_embedding):

        Config.__init__(self, main_task_directory_path, model_path, file_full_word_embedding)

        self.aux_task_data_directory_path       = [path + "/data/" for path in aux_task_directory_path]

        self.file_conll_aux_task_train_data     = [path + "train.data" for path in self.aux_task_data_directory_path]
        self.file_seq_aux_task_train_data       = [path + ".seq" for path in self.file_conll_aux_task_train_data]

