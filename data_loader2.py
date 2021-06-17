from torch.utils.data import Dataset
from utils import load_corpus,normalize_adj
from scipy.sparse import eye
from transformers import BertTokenizer
import torch
import dgl
def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask


class Data_set(Dataset):
    def __init__(self,name):
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
            name)
        doc_mask=train_mask+test_mask+val_mask
        adj=normalize_adj(adj+eye(adj.shape[0]))
        train_num=train_mask.sum().item()
        val_num=val_mask.sum().item()
        test_num=test_mask.sum().item()
        node_size=adj.shape[0]
        y=torch.tensor(y_train+y_val+y_test)
        self.y=torch.argmax(y,-1)

        self.train_index=[i for i in range(train_num+val_num)] + [i for i in range(node_size-test_num,node_size)]
        corpse_file = open('../data/corpus/' + name +'_shuffle.txt').readlines()
        token=BertTokenizer.from_pretrained('bert-base-uncased')
        self.dataset,self.attention_mask=encode_input(corpse_file,token)
        self.attention_mask=torch.tensor(self.attention_mask)
        self.dataset=torch.tensor(self.dataset)
        self.graph=dgl.from_scipy(adj,eweight_name='w')
        self.graph.ndata['label']=self.y
        self.label_num=len(y_train[0])
        self.graph.edata['w']=self.graph.edata['w'].float()
        self.graph.ndata['train_mask']=torch.tensor(train_mask)
        self.graph.ndata['valid_mask']=torch.tensor(val_mask)
        self.graph.ndata['test_mask']=torch.tensor(test_mask)
        self.train_mask=train_mask
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        return self.dataset[item],self.y[self.train_index[item]],self.attention_mask[item],self.train_mask[self.train_index[item]],self.train_index[item]

