import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from A_mnist import load_mnist
from B_train import TwoLayerNet

# 20250615 we have a test, this is a test
# 加载模型参数
with open('model_params.pkl', 'rb') as f:
    params = pickle.load(f)

# 加载数据集
(img_train, t_train), (img_val, t_val) = load_mnist(normalize=True, one_hot_label=False)


# 选择数据集
# num_images = 16
# random_indices = np.random.choice(len(test_images), num_images, replace=False)
# images = train_images[random_indices]
# labels = train_labels[random_indices]
num_images = 16
sequence = np.arange(1, 17)
images = img_val[sequence]
labels = t_val[sequence]

# 可视化图像并显示它们的标签
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {labels[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# 进行前向预测
net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
net.params = params

# 对一张图片进行预测
test_image = img_val[20]
test_label = t_val[20]

# 将test_image转换为包含单个图像的数组
test_image = np.expand_dims(test_image, axis=0)

# 使用已经加载了模型参数的TwoLayerNet对象对测试图像进行前向传播
scores = net.predict(test_image)

# 计算预测结果
prediction = np.argmax(scores)
confidence = scores[0, prediction]

# 绘制图像和概率分布子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 显示图像
ax1.imshow(test_image.reshape(28, 28), cmap='gray')
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

# 对整个测试集进行前向预测
# 初始化准确率和总样本数
accuracy = 0
total_samples = len(img_val)

for i in tqdm(range(total_samples)):
    test_image = img_val[i]
    test_label = t_val[i]

    # 将test_image转换为包含单个图像的数组
    test_image = np.expand_dims(test_image, axis=0)

    # 使用已经加载了模型参数的TwoLayerNet对象对测试图像进行前向传播
    scores = net.predict(test_image)

    # 计算预测结果
    prediction = np.argmax(scores)

    # 判断预测结果是否正确
    if prediction == test_label:
        accuracy += 1

# 计算准确率
accuracy = accuracy / total_samples * 100

print("准确率：", accuracy, "%")
