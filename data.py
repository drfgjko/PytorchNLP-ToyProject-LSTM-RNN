import json
import os
import numpy as np
from gensim.models.word2vec import Word2Vec
from opencc import OpenCC
import pickle


# 导入数据集+拆分+词向量模型

# opencc繁体转简中
# 每个 JSON 文件有1000条诗.

# json->txt
def json_to_txt():
    # OpenCC 对象，繁体中文->简体中文
    converter = OpenCC('t2s')
    txt = './txt_dataset/song.txt'

    with open(txt, 'w') as f:
        f.write('')

    for i in range(100):
        path = './origin_dataset/poet.song.' + str(i * 1000 + 7000) + '.json'

        # 读取json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 追加写入txt
        with open(txt, 'a', encoding='utf-8') as f:
            for item in data:
                # 获取"paragraphs"
                paragraphs = ''.join(map(str, item.get('paragraphs', '')))

                # 单句有5字或7字，保留统一格式的诗
                # 按句号分割
                sentences = [sentence.strip() for sentence in paragraphs.split('。') if sentence.strip()]

                flag = 1
                # 只处理长度为2的诗
                if len(sentences) == 2:
                    for sentence in sentences:
                        # print("sentence:" + sentence)
                        # print("len:" + str(len(sentence)))
                        # 单句有16字，保留统一格式的诗
                        if len(sentence) != 15:
                            flag = 0
                            break
                    if flag == 1:
                        # paragraphs从繁体->简体，并写入到 test.txt 文件中
                        # 一首诗有32个字(包括符号)
                        paragraphs_simplified = converter.convert(paragraphs.strip())
                        f.write(paragraphs_simplified + '\n')


# 拆分(每个字以空格分开) -> 存入'./dataset/song_split.txt'
# ./txt_dataset/song.txt -> json转成的简中txt
def split_poetry(file='./txt_dataset/song.txt'):
    # ./dataset/song_split.txt -> 要存的文件
    txt_split_file = './dataset/song_split.txt'

    # 读的是原始数据
    data = open(file, "r", encoding="utf-8").read()
    # 以空格拆分每个字
    all_date_split = " ".join(data).strip()

    # 拆分后存入'./dataset/song_split.txt' : 以空格拆分了每个字的txt,含'\n'
    with open(txt_split_file, 'w', encoding="utf-8") as f:
        f.write(all_date_split)


# 词向量,获取w1,key_to_index,index_to_key
def train_vec(split_file='./dataset/song_split.txt', org_file='./txt_dataset/song.txt'):
    vec_params_file = 'vec_params.pkl'

    # './txt_dataset/song.txt'不存在，则json->txt，存文件
    if not os.path.exists(org_file):
        json_to_txt()

    # 存在，则直接使用

    # './dataset/song_split.txt'不存在，则进行拆分：存文件+取出以空格拆分的字符串 :不含'\n'
    if not os.path.exists(split_file):
        split_poetry()

    # 如果已经存在“拆分”好了的文件，直接从文件里读取 list
    split_data = open(split_file, 'r', encoding="utf-8").read().split("\n")
    # org_date = open(org_file, 'r', encoding="utf-8").read().split("\n")

    # 还没进行创建词汇表等工作
    if not os.path.exists(vec_params_file):
        # Word2Vec 模型 vector_size = embedding_dim
        model = Word2Vec(vector_size=110, min_count=1, sg=1, hs=0, workers=10)
        # 构建词汇表：收集数据中出现的所有单词，并为它们分配唯一的索引
        model.build_vocab(split_data)
        # model.syn1neg: 这是Word2Vec模型的负采样权重参数。
        # model.wv.key_to_index: 这是一个字典，将词汇表中的单词映射到其对应的索引。
        # model.wv.index_to_key: 这是一个列表

        # 序列化的对象写入到文件->所需的模型参数存储到pickle文件，二进制写入模式打开文件
        pickle.dump((model.syn1neg, model.wv.key_to_index, dict(enumerate(model.wv.index_to_key))), open("vec_params"
                                                                                                         ".pkl",
                                                                                                         "wb"))

        poem_indices = [[model.wv.key_to_index[word] for word in poem.split()] for poem in split_data]
        poem_array = np.array(poem_indices)

        # model.wv.index_to_key:模型中的所有单词列表:单词列表是唯一的->每个单词只会出现一次
        return poem_array, (model.syn1neg, model.wv.key_to_index, dict(enumerate(model.wv.index_to_key)))

    syn1neg, key_to_index, index_to_key = pickle.load(open(vec_params_file, 'rb'))

    poem_indices = [[key_to_index[word] for word in poem.split()] for poem in split_data]
    poem_array = np.array(poem_indices)

    # 如果已经存在该文件，直接读取
    # vec_params_file 里存的是(model.syn1neg, model.wv.key_to_index, model.wv.index_to_key)
    # 二进制读取模式打开文件
    return poem_array, (syn1neg, key_to_index, index_to_key)
