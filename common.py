import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import utils
from collections import Counter
import sentencepiece as spm


# 学習データの単語長の最大値と最小値を設定する
minlen = 5
maxlen = 128

sp_model_filepath = 'wiki-ja.model'
checkpoint_path = 'model.ckpt-1400000'

# sentencepieceのモデルファイル読み込み
sp = spm.SentencePieceProcessor()
sp.Load(sp_model_filepath)

# ハードウェア情報の取得とプロセッサの選択
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")

os.environ['TF_KERAS'] = '1' # keras-bert用の環境変数。TPUで必須（CPU/GPUでも tf.keras を使用するためには必要）
if tpu:
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128) # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
else:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)


# dataframe→inputへの変換用関数
def get_indice(feature):
    indices = np.zeros((maxlen), dtype=np.int32)

    tokens = []
    tokens.append('[CLS]')
    tokens.extend(sp.encode_as_pieces(feature))
    tokens.append('[SEP]')

    if len(tokens) < minlen + 2:
        return None

    for t, token in enumerate(tokens):
        if t >= maxlen:
            break
        try:
            indices[t] = sp.piece_to_id(token)
        except:
            indices[t] = sp.piece_to_id('<unk>')

    return indices


def get_text_label(df, label2index, class_count):
    text, label = [], []

    for i, t, l in df.itertuples():
        idices = get_indice(t)
        if idices is not None:
            text.append(idices)
            label.append(l)

    print(Counter(label))
    label = utils.np_utils.to_categorical([label2index[l] for l in label], num_classes=class_count)
    segments = np.zeros((len(text), maxlen), dtype=np.float32)

    return text, label, segments
