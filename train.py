import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import roc_curve, auc

# 读取数据集并进行预处理
train_data = pd.read_csv('/home/wencaiwang/Proj/data/train.csv')


# [['病毒载量', 'APTT', 'PCT', '天门冬氨酸氨基转移酶', '意识障碍', '尿素氮', '年龄', '肌酐', '单核细胞.']]
train_y = train_data['Target']
train_x = train_data[['病毒载量', 'APTT', 'PCT', '天门冬氨酸氨基转移酶', '意识障碍', '肌酐', '尿素氮','年龄', '单核细胞']]
min_value = train_x.min()
max_value = train_x.max()
train_x = (train_x - min_value) / (max_value - min_value)   # 对训练数据归一化处理


# 定义模型结构
model = Sequential()  # 60 95 50
model.add(Dense(60, input_dim=9, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(95, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 定义训练参数并编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型
model.fit(train_x, train_y, epochs=200, batch_size=512, validation_split=0.2)
#model.fit(train_x, train_y, epochs=10, batch_size=512)

# 模型以及权重保存
model.save('/home/wencaiwang/Proj/MLP_model.h5')
model.predict
