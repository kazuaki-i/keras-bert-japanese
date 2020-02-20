from keras_bert import get_custom_objects
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model

from .common import *

input_model_filename = 'input_model.hdf5'
output_model_filename = 'output_model.hdf5'

# bertモデルの読み込み
model = load_model(input_model_filename, custom_objects=get_custom_objects())

# finetune用に追加したレイヤーを削る（最後のラベル当ての部分を削る）

layer_name = 'NSP-Dense'
model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# レイヤーを削ったモデルを保存
model.save(output_model_filename)
