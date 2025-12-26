from sklearn.preprocessing import MinMaxScaler

# 1,准备数据集
x_train = [[0,1,2],[1,2,3],[2,3,4],[3,4,5]]

# 2,实例化
scaler = MinMaxScaler(feature_range=(1,3))

# 3,归一化操作
x_train_new = scaler.fit_transform(x_train)
# 4,打印结果
print(x_train_new)


