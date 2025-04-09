import pickle
import random
import torch
from opencc import OpenCC
from torch.autograd import Variable
from data import train_vec
from mymodel_lstm import MyLstmModel
from mymodel_rnn import MyRnnModel
from train import get_args, train
import torch.nn.functional as f

device = "cuda" if torch.cuda.is_available() else "cpu"
vec_params_file = 'vec_params.pkl'
org_file = './txt_dataset/song.txt'

# 默认是Test
# 1: 训练lstm 2：训练rnn 3：test
mode = 3


# 加载模型
def load_model(file):
    all_data, (w1, word_2_index, index_2_word) = train_vec()
    _, hidden_dim, num_layers, _, _ = get_args()
    vocab_size, embedding_dim = w1.shape

    if file == './model/model_lstm':
        # MyLstmModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        model = MyLstmModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        # 实例化模型时
        model.load_state_dict(torch.load(file))
        # 设置为评估模式
        model.eval()
        return model

    elif file == './model/model_rnn':
        # MyRnnModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        model = MyRnnModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        # 实例化模型时
        model.load_state_dict(torch.load(file))
        # 设置为评估模式
        model.eval()
        return model


# 输入单字，生成的诗句格式比较规定
def generate_poem(model, input_char, max_length=31):
    (w1, word_2_index, index_2_word) = pickle.load(open(vec_params_file, 'rb'))

    result = []
    x_input = Variable(
        torch.Tensor([word_2_index[input_char]]).view(1, 1).long())

    x_input.to(device)
    model.to(device)

    h_0 = None
    c_0 = None
    word_index = word_2_index[input_char]
    # is_punctuation = False

    # 词汇表里没有这个字
    if word_index == -1:
        return "请换个字输入！"

    # 先把字添加进去
    result.append(input_char)

    if isinstance(model, MyRnnModel):
        for i in range(max_length):
            pre, h_0 = model(x_input, h_0)
            # word_index = int(torch.argmax(pre))
            # top_index = pre.data[0].topk(1)[1][0]
            top_7_probs, top_7_indices = torch.topk(pre[0], 7)

            probs = f.softmax(pre, dim=1)[0]
            top_7_probs = probs[top_7_indices]
            top_index = random.choices(top_7_indices, weights=top_7_probs, k=1)[0]
            pre_word = index_2_word[top_index.item()]

            result.append(pre_word)
            x_input = Variable(x_input.data.new([top_index])).view(1, 1)
        if len(result) < max_length:
            return "请换个字输入！"
        return ''.join(result)

    # 默认使用LSTM -> 同字生成的诗歌可能更多样一些
    for i in range(max_length):
        pre, (h_0, c_0) = model(x_input, h_0, c_0)
        # word_index = int(torch.argmax(pre))
        # top_index = pre.data[0].topk(1)[1][0]

        # ——————————————————————————————————————————————————
        # top_7_probs, top_7_indices = torch.topk(pre[0], 7)
        #
        # if result[-1] in ["，", "。"]:
        #     # print("符号位")
        #     probs = f.softmax(pre, dim=1)[0]
        #     top_7_probs = probs[top_7_indices]
        #     top_index = random.choices(top_7_indices, weights=top_7_probs, k=1)[0]
        #     pre_word = index_2_word[top_index.item()]
        # else:
        #     top_index = pre.data[0].topk(1)[1][0]
        #     pre_word = index_2_word[top_index.item()]
        # ——————————————————————————————————————————————————

        top_7_probs, top_7_indices = torch.topk(pre[0], 7)

        probs = f.softmax(pre, dim=1)[0]
        top_7_probs = probs[top_7_indices]
        top_index = random.choices(top_7_indices, weights=top_7_probs, k=1)[0]
        pre_word = index_2_word[top_index.item()]

        result.append(pre_word)
        x_input = Variable(x_input.data.new([top_index])).view(1, 1)
    if len(result) < max_length:
        return "请换个字输入！"
    return ''.join(result)


# mode = 3 -> test
def test(start_word, model_type=1):
    if model_type == 2:
        # 加载模型并设置为eval
        model = load_model('./model/model_rnn')
        generated_poem = generate_poem(model, start_word)
        return generated_poem
    # 默认 使用LSTM
    model = load_model('./model/model_lstm')
    generated_poem = generate_poem(model, start_word)
    return generated_poem


def is_chinese(words):
    return '\u4e00' <= words <= '\u9fff'


# def generate_random_poem(model, max_length=31, num_poems=5):
#     (w1, word_2_index, index_2_word) = pickle.load(open('vec_params.pkl', 'rb'))
#
#     poems = []
#     for _ in range(num_poems):
#         starting_word = random.choice(list(word_2_index.keys()))
#         poem = generate_poem(model, starting_word, max_length)
#         poems.append(poem)
#
#     return poems

# BLEU类用不了
# def score(type=1):
#     # 默认
#     model = load_model('./model/model_lstm')
#
#     if type == 2:
#         model = load_model('./model/model_rnn')
#
#     # 生成x首随机诗句
#     poems = generate_random_poem(model)
#     # 读取参考诗句
#     with open('./txt_dataset/song.txt', 'r', encoding="utf-8") as f:
#         reference_sentences = f.read().splitlines()
#
#     # print(reference_sentences)
#
#     reference_sentences = [s.split() for s in reference_sentences[:5]]
#     reference_sentences = [reference_sentences]
#
#     candidate_sentences = [s.split() for s in poems]
#
#     # print(poems)
#
#     print(reference_sentences)
#
#     print(candidate_sentences)
#
#     # smoothie = SmoothingFunction()
#     bleu = sacrebleu.BLEU()
#     bleu_score = bleu.corpus_score(candidate_sentences, reference_sentences).score
#     bleu_score = corpus_bleu(reference_sentences, candidate_sentences, smoothing_function=smoothie.method1)
#     print(f"BLEU-4 Score: {bleu_score: .4f}")


if __name__ == '__main__':
    # train(1)
    # train(2)

    # 模型训练
    # train(mode)

    # print(all_data)
    # all_data: numpy数组 (poem_num,poem_words_num )
    # (诗歌的数量，每首诗歌的字数) -> (_,32)
    # print(np.shape(all_data))

    # word_2_index:dict  eg: '字':1
    # print(word_2_index)
    # index_2_word: 转成跟word_2_index相似的字典
    # print(index_2_word)
    # ________________________________
    converter = OpenCC('t2s')
    if mode == 3:
        while True:
            print("请输入一个字:", end='')
            word = input().strip()
            if not word:
                print("输入为空")
                break
            word = converter.convert(word[0])
            if not is_chinese(word):
                print("请输入中文")
                continue
            else:
                # test 默认使用LSTM，设为2则使用RNN
                out_poem = test(word)
                print(out_poem)

    # ________________________________
    # score()
