from torch.utils.data import Dataset
from torch import tensor
from transformers import BertTokenizer
import pickle


class DataSet(Dataset):
    def __init__(self, name, usage, label_dict=None):
        tokenizers = BertTokenizer.from_pretrained('bert-base-uncased')
        # data=open(usage+'_data.txt').readlines()
        # label=open(usage+'_label.txt').readlines()
        # self.data=[tokenizers.encode(each,max_length=512) for each in data]
        # self.label=[int(i) for i in label]
        # self.class_num=23
        label_path = '../data/' + name + '_labels.pkl'
        indexs = '../data/' + name + '_indexs.pkl'
        current_usage = '../data/' + name + '_' + usage + '_index.pkl'
        current_usage = pickle.load(open(current_usage, 'rb'))
        orig_data_path = '../data/corpus/' + name + '.txt'
        orig_data = open(orig_data_path).readlines()
        indexs = pickle.load(open(indexs, 'rb'))
        labels = pickle.load(open(label_path, 'rb'))
        orig_data = [orig_data[i] for i in indexs]
        orig_data = [tensor(tokenizers.encode(each, max_length=512)) for each in orig_data]
        self.data = [orig_data[i] for i in current_usage]
        self.label = [labels[i] for i in current_usage]
        self.class_num = len(pickle.load(open('../data/' + name + '_label_dict.pkl', 'rb')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return tensor(self.data[item]), tensor(self.label[item])
# data=DataSet('ohsumed','train')
