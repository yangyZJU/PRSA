from abc import ABC, abstractmethod
import pandas as pd

class DataProcessor(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.data_dir = self.opt["data_dir"]
        self.theme = self.opt["theme"]
        self.split_size = 3

    @abstractmethod
    def load_data(self):
        pass



class Datasets(DataProcessor):

    def load_data(self):
        df = pd.read_csv(self.data_dir + '/%s.csv'%self.theme, encoding="utf-8")
        exs = df.reset_index().to_dict('records')
        length_of_exs = len(exs)
        size_per_split = length_of_exs // self.split_size
        train_data = exs[:(self.split_size - 1) * size_per_split]
        test_data = exs[(self.split_size - 1) * size_per_split:]
        return train_data, test_data
    
    def load_collect_data(self):
        df = pd.read_csv(self.data_dir + '/%s.csv'%self.theme, encoding="utf-8")
        exs = df.reset_index().to_dict('records')
        return exs
    
    def load_test_data(self):
        df = pd.read_csv(self.data_dir + '/%s.csv'%self.theme, encoding="utf-8")
        exs = df.reset_index().to_dict('records')
        length_of_exs = len(exs)
        test_data = exs[0:length_of_exs]
        return test_data
    
    
def data_batch(data, batch_size):
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return batches






