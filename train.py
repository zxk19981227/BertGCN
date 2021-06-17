from Model import BertGCN
from torch.utils.data import DataLoader
from data_load import data_load
from utils import get_args
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import nll_loss
import torch
from numpy import mean
from torch import log
from tqdm import tqdm
from torch.nn.functional import softmax
from utils import setup_seed
from torch.optim import lr_scheduler
from data_loader2 import Data_set

setup_seed(9)


def train(i, dataset, model: BertGCN, optim, features, graph, device):
    model.train()
    losses = []
    correct = 0
    total = 0
    increase=0
    for src, label, attention,mask, index in tqdm(dataset):
        mask = mask.to(device)
        src = src.to(device)
        attention=attention.to(device)
        label = label.to(device)
        predict = model(src, features,attention, graph, index)
        predict = predict[mask]
        label = label[mask]
        if predict.shape[0] == 0:
            continue
        loss = nll_loss(log(predict), label)
        loss.backward(retain_graph=True)
        increase+=1
        if increase%4==0:
            optim.step()
            optim.zero_grad()
            increase=0
        total += predict.shape[0]
        correct += (torch.argmax(predict, -1) == label).sum().item()
        losses.append(loss.item())
    print("training epoch {} || loss {} || accuracy {}".format(i, mean(losses), correct / total))

#
# def collate_fn(batch):
#     src, trg, mask, idx = [], [], [], []
#     atts=[]
#     for s, t,att, m, i in batch:
#         src.append(s)
#         trg.append(t)
#         mask.append(m)
#         idx.append(i)
#     src = pad_sequence(src, batch_first=True)
#     trg = torch.tensor(trg)
#     mask = torch.tensor(mask) == 1
#     idx = torch.tensor(idx)
#     return src, trg, mask, idx


def update_features(features, dataset, model, device):
    with torch.no_grad():
        model.eval()
        for src, label,attention, mask, idx in tqdm(dataset):
            src = src.to(device)
            attention=attention.to(device)
            current_features = model.BertModel.model(src,attention_mask=attention).last_hidden_state[:, 0, :]
            features[idx] = current_features.detach()
    return features


def eval(i, dataset, model: BertGCN, features, graph, usage, device, best_loss=None, best_accuracy=None,
         no_increasing=None):
    model.eval()
    if usage == 'valid':
        mask = graph.ndata['valid_mask']
    else:
        mask = graph.ndata['test_mask']
    mask = (mask == 1)
    if usage == 'test':
        model.load_state_dict(torch.load('best_Bert_GCN_model.pkl'))
        features = update_features(features, dataset, model, device)
    predict = model.BertModel.linear(features)
    graph_predict = model.gcn(graph, features)
    predict = softmax(predict[mask], -1) * (1 - model.lam) + softmax(graph_predict[mask], -1) * model.lam
    label = graph.ndata['label']
    loss = nll_loss(log(predict), label[mask])
    correct = (torch.argmax(predict, -1) == label[mask]).sum().item()
    total = sum(mask).item()
    print("{} epoch loss {} || accuracy {}".format(usage, loss.item(), correct / total))
    if usage == 'test':
        model.load_state_dict(torch.load('best_accuracy.pkl'))
        features = update_features(features, dataset, model, device)
        predict = model.BertModel.linear(features)
        graph_predict = model.gcn(graph, features)
        predict = softmax(predict[mask], -1) * (1 - model.lam) + softmax(graph_predict[mask], -1) * model.lam
        label = graph.ndata['label']
        loss = nll_loss(log(predict), label[mask])
        correct = (torch.argmax(predict, -1) == label[mask]).sum().item()
        total = sum(mask).item()
        print("best_accuracy{}  epoch loss {} || accuracy {}".format(usage, loss.item(), correct / total))
    if usage == 'valid':
        if best_loss > loss.item():
            best_loss = loss.item()
            no_increasing = 0
            torch.save(model.state_dict(), 'best_Bert_GCN_model.pkl')
            print("saving to file best_Bert_GCN_model.pkl")

        else:
            no_increasing += 1
        if best_accuracy < correct / total:
            best_accuracy = correct / total
            torch.save(model.state_dict(), 'best_accuracy.pkl')
            print("saving to file best_accuracy.pkl")
        return best_loss, best_accuracy, no_increasing


def main():
    args = get_args()
    device = 'cuda:0'
    dataset = Data_set(args.dataset)
    graph = dataset.graph.to(device)
    data_loader = DataLoader(dataset,  batch_size=16, shuffle=True)
    features = torch.zeros(graph.num_nodes(), 768, requires_grad=False).to(device)
    model = BertGCN('./best_pretrain_bert.pkl', dataset.label_num)
    model = model.to(device)
    optim = torch.optim.Adam(
        [{'params': model.gcn.parameters(), 'lr': 1e-3}, {'params': model.BertModel.parameters(), 'lr': 1e-5}])
    scheduler=lr_scheduler.MultiStepLR(optim,milestones=[30],gamma=0.1)
    features = update_features(features, data_loader, model, device)
    best_loss = 1e10
    no_increasing = 0
    best_accuracy = 0
    for i in range(20):
        train(i, data_loader, model, optim, features, graph, device)
        scheduler.step()
        torch.cuda.empty_cache()
        with torch.no_grad():
            features = update_features(features, data_loader, model, device)
            best_loss, best_accuracy, no_increasing = eval(i, data_loader, model, features, graph, 'valid', device,
                                                           best_loss,
                                                           best_accuracy,
                                                           no_increasing)
        if no_increasing >= 10:
            break  # for i in range(20):
    with torch.no_grad():
        eval(0, data_loader, model, features, graph, 'test', device, best_loss, best_accuracy, no_increasing)


if __name__ == '__main__':
    main()
