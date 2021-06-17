from transformers import BertModel, BertConfig
from torch.nn import Module, Linear
from BertConfig import Config


class finetunedBert(Module):
    def __init__(self, class_num):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.linear = Linear(768, class_num)

    def forward(self, input):
        predict = self.model(input).last_hidden_state[:,0,:]
        return self.linear(predict)
