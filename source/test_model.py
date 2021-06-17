from pretrain_bert import finetunedBert
import argparse
from torch.optim import AdamW
from utils import get_args
from pre_train_data_load import DataSet
import pickle
from torch.nn.functional import cross_entropy
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from numpy import mean
from tqdm import tqdm
from Model import BertGCN
from data_load import data_load


def collate_fn(batch):
    src, label, masks, indexs = [], [], [], []
    for s, t, mask, index in batch:
        src.append(s)
        label.append(t)
        masks.append(mask)
        indexs.append(index)
    src = pad_sequence(src, batch_first=True, padding_value=0)
    label = torch.tensor(label)
    masks = torch.tensor(masks)
    indexs = torch.tensor(indexs)
    return src, label, masks, indexs


def train(i, model, optim, data_loader, device):
    model.train()
    losses = []
    correct = 0
    total = 0
    for src, trg in tqdm(data_loader):
        optim.zero_grad()
        src = src.to(device)
        trg = trg.to(device)
        predict = model(src)
        loss = cross_entropy(predict, trg.long())
        loss.backward()
        optim.step()
        losses.append(loss.item())
        correct += (torch.argmax(predict, -1) == trg).sum().item()
        total += predict.shape[0]
    print("train epoch {} accuracy {} || loss {}".format(i, correct / total, mean(losses)))


def eval(i, model, best_loss, no_increase, data_loader, device):
    model.eval()
    losses = []
    correct = 0
    total = 0
    for src, trg in tqdm(data_loader):
        src = src.to(device)
        trg = trg.to(device)
        predict = model(src)
        loss = cross_entropy(predict, trg.long())
        losses.append(loss.item())
        correct += (torch.argmax(predict, -1) == trg).sum().item()
        total += predict.shape[0]
    loss = mean(losses)
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), 'best_pretrain_bert.pkl')
        no_increase = 0
    else:
        no_increase += 1
    print("eval epoch {} accuracy {} || loss {}".format(i, correct / total, mean(losses)))
    return best_loss, no_increase


def test(model, data_loader, device):
    model.eval()
    losses = []
    correct = 0
    total = 0
    for src, trg, mask, idx in tqdm(data_loader):
        src = src.to(device)
        trg = trg.to(device)
        predict = model(src)
        loss = cross_entropy(predict, trg.long())
        losses.append(loss.item())
        correct += (torch.argmax(predict, -1) == trg).sum().item()
        total += predict.shape[0]
    loss = mean(losses)
    print("test accuracy {} || loss {}".format(correct / total, mean(losses)))


def update_features(features, dataset, model, device):
    with torch.no_grad():
        model.eval()
        for src, label, mask, idx in tqdm(dataset):
            src = src.to(device)
            current_features = model.BertModel.model(src).last_hidden_state[:, 0, :]
            features[idx] = current_features
    return features


def pretrain():
    args = get_args()
    device = 'cuda:1'
    data = DataSet(args.dataset, 'train')
    train_loader = DataLoader(data, collate_fn=collate_fn, batch_size=20, shuffle=True)
    valid_data = DataSet(args.dataset, 'valid')
    val_loader = DataLoader(valid_data, collate_fn=collate_fn, batch_size=20)
    test_data = data_load(args.dataset)
    test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=20)
    # word_num=pickle.load(open('../data/'+args.dataset+'_dict.pkl','rb'))
    model = BertGCN('best_pretrain_bert.pkl', data.class_num)

    model = model.to(device)
    optim = AdamW(model.parameters(), lr=2e-5)
    best_loss = 1e10
    no_increasing = 0
    # for i in range(10):
    #     train(i, model, optim, train_loader, device)
    #     with torch.no_grad():
    #         no_increasing, best_loss = eval(i, model, best_loss, no_increasing, val_loader, device)
    #         if no_increasing>10:
    #             break
    # model.load_state_dict(torch.load('best_pretrain_bert.pkl'))
    with torch.no_grad():
        test(model, test_loader, device)


pretrain()
