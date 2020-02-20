# ハイパーパラメータの設定
from keras_bert import get_custom_objects
from tensorflow.python.keras.models import load_model
import sentencepiece as spm
import numpy as np

from .common import *

# 入力文字列の最大長（これ以上長いものはカットされる）
# 適切な長さを設定すること

input_model_filename = 'input.hdf5'
model = load_model(input_model_filename, custom_objects=get_custom_objects())


def texts2matrix(texts):
    '''
    input
      texts: input batch of texts' list ([text, text, text, ...])
    output
      matrix of texts (numpy.array, dim=bert NSP-Dense)
    '''

    common_seg_input = np.zeros((len(texts), maxlen), dtype=np.float32)
    matrix = np.zeros((len(texts), maxlen), dtype=np.float32)

    for i, text in enumerate(texts):
        sp_lst = [t for t in sp.encode_as_pieces(text)]

        if not sp_lst:
            print('text is emputy. skip convert matrix', sp_lst)
            continue

        tokens = []
        tokens.append('[CLS]')
        tokens.extend(sp_lst[:maxlen])
        if len(sp_lst) <= maxlen:
            tokens.append('[SEP]')

        for j, token in enumerate(tokens):
            try:
                matrix[i, j] = sp.piece_to_id(token)
            except:
                matrix[i, j] = sp.piece_to_id('<unk>')

    return model.predict([matrix, common_seg_input])


# 入力された2つの文字列のコサイン類似度を計算する
def bert_cos_sim(texts1, texts2):
    '''
    input
      texts1, texts2: batch texts ([texts, texts, ...])
    output:
      None (ただし、printが実行される)
    '''

    if len(texts1) != len(texts2):
        raise ValueError('input length is different')

    matrix1 = texts2matrix(texts1)
    matrix2 = texts2matrix(texts2)

    for t1, t2, m1, m2 in zip(texts1, texts2, matrix1, matrix2):
        print('input1: {}\ninput2: {}'.format(t1, t2))
        v1 = m1.flatten()
        v2 = m2.flatten()
        cossim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        print('cos sim: {}\n'.format(cossim))


# 単語分割
l0 = [
    '走る走る俺たち 流れる汗もそもままに。',
    'いつかたどり着いたら、きっと君に打ち明けられるだろう。'
]

for text in l0:
    print(sp.encode_as_pieces(text))

# 類似度計算
l1 = ['走る走る俺たち 流れる汗もそもままに。']
l2 = ['いつかたどり着いたら、きっと君に打ち明けられるだろう。']

bert_cos_sim(l1, l2)
