import numpy as np
import gradio as gr
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from keras.models import load_model
#加载模型以及模型所需基础变量

train_data = pd.read_csv('/home/wencaiwang/Proj/data/train.csv')
train_y = train_data['Target']
train_x = train_data[['病毒载量', 'APTT', 'PCT', '天门冬氨酸氨基转移酶', '意识障碍', '肌酐', '尿素氮','年龄', '单核细胞']]
min_value = train_x.min()
max_value = train_x.max()
print(type(min_value))

# 加载模型
loaded_model_DL = tf.keras.models.load_model('/home/wencaiwang/Proj/MLP_model.h5')

# 定义预测函数

def predict1(var1, var2, var3, var4, var5, var6, var7, var8,var9):
    # 将输入的9个属性数据组合成 DataFrame 对象
    data = {'病毒载量': [var1], 'APTT': [var2], 'PCT': [var3], '天门冬氨酸氨基转移酶': [var4], '意识障碍': [var5], '肌酐': [var6], '尿素氮': [var7], '年龄': [var8], '单核细胞': [var8]}
    data = pd.DataFrame(data)
    print(type(data))
    data =(data - min_value) / (max_value - min_value)
    DL_pred_y = loaded_model_DL.predict(data)
    return DL_pred_y


# 定义9个属性的输入框
inputs = [
    gr.Number(label="病毒载量"),
    gr.Number(label="APTT"),
    gr.Number(label="PCT"),
    gr.Number(label="天门冬氨酸氨基转移酶"),
    gr.Number(label="意识障碍"),
    gr.Number(label="肌酐"),
    gr.Number(label="尿素氮"),
    gr.Number(label="年龄"),
    gr.Number(label="单核细胞")
]

# 创建输出组件
output1 = gr.Number(label="DL")

# 创建界面
interface = gr.Interface(fn=predict1, inputs=inputs, outputs=output1)
# 创建 Gradio 界面
#interface = gr.Interface(fn=predict, inputs=inputs, outputs="number")

# 运行 Gradio 界面
interface.launch(share=True)