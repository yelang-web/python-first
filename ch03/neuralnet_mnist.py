# coding: utf-8
import os
import sys

sys.path.append('/home/wuyanzu/workspace/python/python-first')  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
	(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return x_test, t_test


def init_network():
	file_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的目录
	weight_path = os.path.join(file_path, "sample_weight.pkl")  # 构建权重文件的完整路径
	with open(weight_path, 'rb') as f:
		network = pickle.load(f)
	return network


def predict(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']
	
	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = softmax(a3)
	
	return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
	y = predict(network, x[i])
	p = np.argmax(y)  # 获取概率最高的元素的索引
	if p == t[i]:
		accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
