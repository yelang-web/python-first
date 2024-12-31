# coding: utf-8
try:
	import urllib.request
except ImportError:
	raise ImportError('You should use Python 3.x')
import gzip
import os
import os.path
import pickle
import time

import numpy as np

url_base = ' https://github.com/golbin/TensorFlow-MNIST/raw/refs/heads/master/mnist/data/'
key_file = {
	'train_img': 'train-images-idx3-ubyte.gz',
	'train_label': 'train-labels-idx1-ubyte.gz',
	'test_img': 't10k-images-idx3-ubyte.gz',
	'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
	file_path = dataset_dir + "/" + file_name
	
	if os.path.exists(file_path):
		return
	
	print("Downloading " + file_name + " ... ")
	try:
		# 添加错误处理和重试逻辑
		max_retries = 3
		for attempt in range(max_retries):
			try:
				headers = {
					'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
				}
				req = urllib.request.Request(url_base + file_name, headers=headers)
				response = urllib.request.urlopen(req, timeout=30)
				
				with open(file_path, 'wb') as f:
					f.write(response.read())
				break
			except urllib.error.URLError as e:
				if attempt == max_retries - 1:
					raise
				print(f"尝试 {attempt + 1} 失败，正在重试...")
				time.sleep(1)
	except Exception as e:
		print(f"下载失败: {str(e)}")
		print("请尝试手动下载数据集并放置在正确的目录中")
		raise
	print("Done")


def download_mnist():
	for v in key_file.values():
		_download(v)


def _load_label(file_name):
	file_path = dataset_dir + "/" + file_name
	
	print("Converting " + file_name + " to NumPy Array ...")
	with gzip.open(file_path, 'rb') as f:
		labels = np.frombuffer(f.read(), np.uint8, offset=8)
	print("Done")
	
	return labels


def _load_img(file_name):
	file_path = dataset_dir + "/" + file_name
	
	print("Converting " + file_name + " to NumPy Array ...")
	with gzip.open(file_path, 'rb') as f:
		data = np.frombuffer(f.read(), np.uint8, offset=16)
	data = data.reshape(-1, img_size)
	print("Done")
	
	return data


def _convert_numpy():
	dataset = {}
	dataset['train_img'] = _load_img(key_file['train_img'])
	dataset['train_label'] = _load_label(key_file['train_label'])
	dataset['test_img'] = _load_img(key_file['test_img'])
	dataset['test_label'] = _load_label(key_file['test_label'])
	
	return dataset


def init_mnist():
	download_mnist()
	dataset = _convert_numpy()
	print("Creating pickle file ...")
	with open(save_file, 'wb') as f:
		pickle.dump(dataset, f, -1)
	print("Done!")


def _change_one_hot_label(X):
	T = np.zeros((X.size, 10))
	for idx, row in enumerate(T):
		row[X[idx]] = 1
	
	return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
	"""读入MNIST数据集
	
	Parameters
	----------
	normalize : 将图像的像素值正规化为0.0~1.0
	one_hot_label :
		one_hot_label为True的情况下，标签作为one-hot数组返回
		one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
	flatten : 是否将图像展开为一维数组
	
	Returns
	-------
	(训练图像, 训练标签), (测试图像, 测试标签)
	"""
	if not os.path.exists(save_file):
		init_mnist()
	
	with open(save_file, 'rb') as f:
		dataset = pickle.load(f)
	
	if normalize:
		for key in ('train_img', 'test_img'):
			dataset[key] = dataset[key].astype(np.float32)
			dataset[key] /= 255.0
	
	if one_hot_label:
		dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
		dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
	
	if not flatten:
		for key in ('train_img', 'test_img'):
			dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
	
	return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
	init_mnist()
