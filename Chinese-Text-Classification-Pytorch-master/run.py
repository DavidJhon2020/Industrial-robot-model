# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import torch.nn.functional as F
import flask
import flask.app
from flask import app, Flask
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='TextCNN',
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')

parser.add_argument('--server', default=True, type=bool, help='True for server, False for train/valid/test')

args = parser.parse_args()

dataset = 'THUCNews'  # 数据集

# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
embedding = 'embedding_SougouNews.npz'
if args.embedding == 'random':
    embedding = 'random'
model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
if model_name == 'FastText':
    from utils_fasttext import build_dataset, build_iterator, get_time_dif

    embedding = 'random'
else:
    from utils import build_dataset, build_iterator, get_time_dif

x = import_module('models.' + model_name)
config = x.Config(dataset, embedding)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()
print("Loading data...")
vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# train
config.n_vocab = len(vocab)
model = x.Model(config).to(config.device)
if model_name != 'Transformer':
    init_network(model)

print(model.parameters)

if args.server:
    config.batch_size = 1
    class_dict = config.class_list
    model.load_state_dict(torch.load(config.save_path))
else:

    train(config, model, train_iter, dev_iter, test_iter)

if args.word:
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
else:
    tokenizer = lambda x: [y for y in x]  # char-level

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin'] = '*'
    environ.headers['Access-Control-Allow-Method'] = '*'
    environ.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return environ


@app.route('/online_test/<sentence>', methods=['get', 'post'])
def online_test(sentence):
    print(class_dict)

    lin = sentence.strip()
    if not lin:
        return
    model.eval()
    with torch.no_grad():
        contents = []
        words_line = []
        token = tokenizer(lin)
        seq_len = len(token)
        if config.pad_size:
            if len(token) < config.pad_size:
                token.extend([PAD] * (config.pad_size - len(token)))
            else:
                token = token[:config.pad_size]
                seq_len = config.pad_size
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        contents.append((words_line, 0, seq_len))

        iter = build_iterator(contents, config)
        for texts, _ in iter:
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()

        type_num = predic[0]
    print(type_num)
    # return jsonify({'类别': class_dict[type_num]})
    return class_dict[type_num]

if __name__ == '__main__':
    app.run(debug=True)
