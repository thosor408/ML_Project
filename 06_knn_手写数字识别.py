import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter
# 1. 显示图片.
def show_digit(idx):
    # 1.1 加载数据.
    data = pd.read_csv('data/手写数字识别.csv')
    # 1.2非法值校验.
    if idx < 0 or idx > len(data) - 1:
        return
    # 1.3 打印数据基本信息
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    print(f'数据基本信息: {x.shape})')
    print(f'类别数据比例: {Counter(y)}')

    # 显示图片
    # 1.4 将数据形状修改为: 28*28
    digit = x.iloc[idx].values.reshape(28, 28)
    # 1.5 关闭坐标轴标签
    plt.axis('off')
    # 1.6 显示图像
    plt.imshow(digit, cmap='gray')  # 灰色显示
    plt.show()


# 2. 训练模型.
def train_model():
    # 1. 加载数据.
    data = pd.read_csv('data/手写数字识别.csv')
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    # 2.数据预处理, 归一化.
    x = x / 255

    # 3. 分割训练集和测试集.
    # stratify: 按照y的类别比例进行分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=21)

    # 4. 训练模型
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5. 模型评估
    my_score = estimator.score(x_test, y_test)
    print(f'测试集准确率为: {my_score:.2f}')

    # 6. 模型保存.
    joblib.dump(estimator, 'my_model/knn.pth')

# 3. 测试模型.
def use_model():
    # 1. 读取图片
    img = plt.imread('data/demo.png')   # 灰度图, 28*28像素
    plt.imshow(img, cmap='gray')
    plt.show()

    # 2. 加载模型.
    estimator = joblib.load('my_model/knn.pth')

    # 3. 预测图片.
    img = img.reshape(1, -1) # 形状从: (28, 28) => (1, 784)
    # print(img.shape)
    y_test = estimator.predict(img)
    print(f'您绘制的数字是: {y_test}')

# 在main函数中测试
if __name__ == '__main__':
    # 1. 调用函数, 查看图片.
    # show_digit(0)
    # show_digit(10)
    # show_digit(100)

    # 2. 训练模型.
    # train_model()

    # 3. 测试模型
    use_model()