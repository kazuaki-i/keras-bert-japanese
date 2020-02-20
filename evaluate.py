# ベストモデルの読み込み（ベストを手動で選択）
import os
import pandas as pd
from keras_bert import get_custom_objects
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model
from .common import *
import random
from sklearn.metrics import classification_report, confusion_matrix
from operator import itemgetter

best_model_path = ''
model_filename = os.path.join(best_model_path)
best_model = load_model(model_filename, custom_objects=get_custom_objects())


# 評価データの呼び出し
dataset_dir = './'
train_df = pd.read_csv(os.path.join(dataset_dir, 'train.tsv'), sep='\t')
label2index = {k: i for i, k in enumerate(train_df['label'].unique())}
index2label = {i: k for i, k in enumerate(train_df['label'].unique())}
class_count = len(label2index)


pred_df = pd.read_csv(os.path.join(dataset_dir, 'test.tsv'), sep='\t')
pred_text, pred_label, pred_segments = get_text_label(pred_df, label2index, class_count)

print(len(pred_label))

# # (オプション)全部使うと多すぎる場合があるので、適当にサンプリングする
# sample_count = 1000
# idx = sorted(random.sample(range(0, len(pred_text)), min([len(pred_text), sample_count])))
#
# sample_pred_text = [pred_text[i] for i in idx]
# sample_pred_label = pred_label[idx, :]
# sample_pred_segments = pred_segments[idx, :]
#
# print(len(sample_pred_label))


# ラベル予測
prediction = best_model.predict([pred_text, pred_segments], batch_size=128, verbose=1)
# prediction = best_model.predict([sample_pred_text, sample_pred_segments], batch_size=512, verbose=1)

# 評価結果出力

prediction_labels = prediction.argmax(axis=1)
numeric_labels = np.array(pred_label).argmax(axis=1)
# numeric_labels = np.array(sample_pred_label).argmax(axis=1)

target_label_idx = [str(v) for k, v in sorted(index2label.items(), key=itemgetter(0))]

print(classification_report(numeric_labels, prediction_labels, target_names=target_label_idx))
