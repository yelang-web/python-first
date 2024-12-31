# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("mnist",
                                              os.path.join(os.path.dirname(__file__), "..", "dataset", "mnist.py"))
mnist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mnist)

from PIL import Image


def img_show(img):
	pil_img = Image.fromarray(np.uint8(img))
	pil_img.show()


(x_train, t_train), (x_test, t_test) = mnist.load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
