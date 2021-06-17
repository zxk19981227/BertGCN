import argparse
from random import shuffle
import copy
import pickle
from tqdm import tqdm
import collections
import math
from scipy.sparse import coo_matrix


def convert_data_to_index(context):
    dictionary = {}
    index_context = []
    context_length = len(context)
    for line in context:
        line = line.strip().lower().split(' ')
        for word in line:
            if word not in dictionary.keys():
                dictionary[word] = len(dictionary.keys()) + context_length
        index_context.append([dictionary[each] for each in line])
    return dictionary, index_context


def word_window_num(windows):
    single_word_fluency = collections.defaultdict(int)
    tuple_word_fluency = collections.defaultdict(int)
    for window in tqdm(windows):
        current_appear = set()
        for i in range(len(window)):
            if window[i] not in current_appear:
                single_word_fluency[window[i]] += 1
                current_appear.add(window[i])
            for j in range(i + 1, len(window)):
                str1=str(window[i])+','+str(window[j])
                str2=str(window[j])+','+str(window[i])
                if window[i] == window[j]:
                    continue
                if str1 in current_appear or str2 in current_appear:
                    continue
                else:
                    tuple_word_fluency[str1] += 1
                    tuple_word_fluency[str2] += 1
                    current_appear.add(str1)
                    current_appear.add(str2)
    return single_word_fluency, tuple_word_fluency


def word_document(indexed_document, start, end, weight):
    word_document = collections.defaultdict(int)
    for document in indexed_document:
        appear = set()
        for word in document:
            if word not in appear:
                word_document[word] += 1
                appear.add(word)
    for i in range(len(indexed_document)):
        word_dict = collections.defaultdict(int)
        line = indexed_document[i]
        for word in line:
            word_dict[word] += 1
        for key in word_dict.keys():
            start.append(i)
            end.append(key)
            TF = word_dict[key] / len(indexed_document[i])
            IDF = math.log(len(indexed_document) / word_document[key])
            weight.append(TF * IDF)
            start.append(key)
            end.append(i)
            weight.append(TF*IDF)

    return start, end, weight


def build_graph(index_context: list, window_size):
    windows = []
    for line in tqdm(index_context):
        if len(line) <= window_size:
            windows.append(line)
        else:
            for i in range(len(line) - window_size + 1):
                windows.append(line[i:i + window_size])
    print('generating word relations')
    single_word_fluency, tuple_word_fluency = word_window_num(windows)
    window_num = len(windows)
    start = []
    end = []
    weight = []
    for sen in tuple_word_fluency.keys():
        s,t=sen.split(',')
        s,t=int(s),int(t)
        score = math.log(tuple_word_fluency[str(s)+','+str(t)] / window_num / (
                single_word_fluency[s] / window_num * single_word_fluency[t] / window_num))
        if score < 0:
            continue
        start.append(s)
        end.append(t)
        weight.append(score)
    start, end, weight = word_document(index_context, start, end, weight)
    return start, end, weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, default='ohsumed')
    args = parser.parse_args()
    file_path = '../data/' + args.dataset + '.txt'
    with open(file_path) as f:
        lines = f.readlines()
    titles = lines
    orig_titles = copy.deepcopy(titles)
    content_path = '../data/corpus/' + args.dataset + '.clean.txt'
    content = open(content_path).readlines()
    shuffle(titles)
    indexs = [orig_titles.index(each) for each in titles]
    content=[content[i] for i in indexs]
    dictonary, index_data = convert_data_to_index(content)
    train_index = []
    test_index = []
    label_dict = {}
    labels = []
    for i in range(len(indexs)):
        line = titles[i].strip().split()
        if line[1] == 'training':
            train_index.append(i)
        else:
            test_index.append(i)
        label = line[-1]
        if label not in label_dict.keys():
            label_dict[label] = len(label_dict)
        labels.append(label_dict[label])
    shuffle(train_index)
    valid_index = train_index[int(len(train_index) * 0.9):]
    train_index = train_index[:int(len(train_index) * 0.9)]
    start, end, weight = build_graph(index_data, 20)
    matrix = coo_matrix((weight, (start, end)))

    pickle.dump(matrix, open('../data/' + args.dataset + '_matrix.pkl', 'wb'))
    pickle.dump(indexs, open('../data/' + args.dataset + '_indexs.pkl', 'wb'))
    pickle.dump(train_index, open('../data/' + args.dataset + '_train_index.pkl', 'wb'))
    pickle.dump(test_index, open('../data/' + args.dataset + '_test_index.pkl', 'wb'))
    pickle.dump(valid_index, open('../data/' + args.dataset + '_valid_index.pkl', 'wb'))
    pickle.dump(labels, open('../data/' + args.dataset + '_labels.pkl', 'wb'))
    pickle.dump(dictonary, open('../data/' + args.dataset + '_dict.pkl', 'wb'))
    pickle.dump(label_dict, open('../data/' + args.dataset + '_label_dict.pkl', 'wb'))


main()
