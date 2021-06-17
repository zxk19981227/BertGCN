import dgl
import torch
from pretrain_bert import finetunedBert
from torch.nn import Module, Dropout
from dgl.nn.pytorch import GraphConv
from torch.nn.functional import relu, softmax


class GCN(Module):
    def __init__(self, class_num):
        super().__init__()
        self.Conv1 = GraphConv(768, 200, weight=True, activation=relu)
        self.Conv2 = GraphConv(200, class_num, weight=True)
        self.dropout=Dropout(0.1)
        self.dropout2=Dropout(0.2)


    def forward(self, graph: dgl.graph, feature: torch.tensor):
        predict1 = self.dropout(self.Conv1(graph, feature, edge_weight=graph.edata['w']))
        predict2 = self.dropout2(self.Conv2(graph, predict1, edge_weight=graph.edata['w']))
        return predict2


class BertGCN(Module):
    def __init__(self, pretrained_path, label_size, lam=0.3):
        """

        :param pretrained_path: path to pretrained bert model
        """
        super().__init__()
        self.BertModel = finetunedBert(label_size)
        self.BertModel.load_state_dict(torch.load(pretrained_path))
        self.gcn = GCN(label_size)
        self.lam = lam

    def forward(self, sentences, features,attention, graph, indexs):
        last_predict = self.BertModel.model(sentences,attention_mask=attention).last_hidden_state[:, 0, :]
        features[indexs] = last_predict.detach()
        gcn_predict = self.gcn(graph, features)
        bert_predict = self.BertModel.linear(last_predict)
        predict = softmax(gcn_predict[indexs], -1) * self.lam + (1 - self.lam) * softmax(bert_predict, -1)
        return predict
