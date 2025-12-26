# 1.工具包
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# from sklearn.neighbors import KNeighborsRegressor

# 2.数据(特征工程)
# 分类
x = [[0,2,3],[1,3,4],[3,5,6],[4,7,8],[2,3,4]]
y = [0,0,1,1,0]
#x = [[0,1,2],[1,2,3],[2,3,4],[3,4,5]]
#y = [0.1,0.2,0.3,0.4]

# 3.实例化
model =KNeighborsClassifier(n_neighbors=3)
# model =KNeighborsRegressor(n_neighbors=3)

# 4.训练
model.fit(x,y)

# 5.预测
print(model.predict([[4,4,5]]))
