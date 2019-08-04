from common.config import Config

class MTLConfig(Config):

    def __init__(self, main_task_directory_path, aux_task_directory_path):

        Config.__init__(self, main_task_directory_path)

        self.aux_task_directory_path            = aux_task_directory_path + "/"
        self.aux_task_data_directory_path       = self.aux_task_directory_path + "data/"

        self.file_conll_aux_task_train_data     = self.aux_task_data_directory_path + "train.data"
        self.file_seq_aux_task_train_data          = self.file_conll_aux_task_train_data + ".seq"

