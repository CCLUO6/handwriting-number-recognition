import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array
from A_mnist import load_mnist
from B_train import TwoLayerNet

# 加载模型参数
with open('model_params.pkl', 'rb') as f:
    params = pickle.load(f)

# 进行前向预测
net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
net.params = params

# 中文路径
image_path = 'test_1.png'

# 读取图片
image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

# 调整图片大小为28x28像素
resized_image = cv2.resize(image, (28, 28))

# 归一化
inverted_image = resized_image.astype('float32') / 255

# 显示调整后的图片
plt.imshow(inverted_image, cmap='gray')
plt.axis('off')
plt.show()

# 调整形状
test_image = inverted_image.reshape((1, 28, 28, 1))

# 将图像转换为包含单个图像的数组
test_image = np.expand_dims(test_image.flatten(), axis=0)

# 使用已经加载了模型参数的TwoLayerNet对象对测试图像进行前向传播
scores = net.predict(test_image)

# 计算预测结果
prediction = np.argmax(scores)
confidence = scores[0, prediction]

# 绘制图像和概率分布子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 显示图像
ax1.imshow(inverted_image, cmap='gray')
ax1.axis('off')
ax1.set_title('Test Image')

# 显示概率分布
labels = range(10)
ax2.bar(labels, scores[0])
ax2.set_xticks(labels)
ax2.set_xlabel('Class')
ax2.set_ylabel('Probability')
ax2.set_title('Probability Distribution')

plt.tight_layout()
plt.show()

print("预测结果：", prediction)
print("置信度：", confidence)
