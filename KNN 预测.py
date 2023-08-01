import random
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class MyKNN:

    def __init__(self, n_neighbors):
        '''指定近邻个数'''
        self.n_neighbors = n_neighbors  # 近邻数
        # self.x_new = [] # 预测数据集

    def predict(self, x_new):
        '''预测模型, 更新近邻序列和得出测试数据所属类'''
        targetList = []  # 存储预测数据的预测所属类
        dists=[]         #存储预测数据到测试数据的距离  
        for i in range(len(x_new)):
            tp = x_new[i]
            # print(tp)
            distList = self.get_deftDist(tp)  # 获取此时测试数据x_new[i]与预先序列的距离
            for j in range(len(self.x_train)):
                Lmax = max(distList)
                L = 0
                for k in range(len(self.x_train[j])):
                    L += (tp[k] - self.x_train[j][k]) ** 2  # 遍历训练集，计算当前x_new[i]训练集与x_train[j]的距离
                L = math.sqrt(L)
                if L >= Lmax:
                    continue  # 若当前所得距离大于预先序列距离的最大值，进入下一个训练数据
                else:
                    # 更新预先序列中的距离和对应训练数据的id
                    if j not in self.idList:
                        index = distList.index(Lmax)  # 获取最大预先距离的索引
                        self.idList[index] = j  # 换为当前训练数据的id
                        distList[index] = L  # 更新最大预先距离
                # distList[i].append(L)
            # return sorted(distList)
            dists.append(distList)
            flag = [0, 0, 0]
            for m in range(len(self.idList)):  # 遍历预先序列中，计算其中的多数类，判断测试数据属于哪类
                indI = self.idList[m]
                targetI = self.y_train[indI]
                if targetI == 0:
                    flag[0] += 1
                elif targetI == 1:
                    flag[1] += 1
                elif targetI == 2:
                    flag[2] += 1
                # print(flag)
            # print('测试数据所属类: ', flag.index(max(flag)))
            targetList.append(flag.index(max(flag)))  # 返回第一个极大值的索引值
            # print('the distance:\n{}'.format(self.distList))
        return targetList,dists

    def get_trainData(self, x_train, y_train):
        '''获取训练集数据'''
        self.x_train = x_train
        self.y_train = y_train
        self.idList = random.sample(range(0, len(x_train)), self.n_neighbors)
        ''' 获取空间大小为k的预先序列,k个随机的元         	
        组,k=n_neighbors'''
        # self.deftDic = {'id':self.idList, 'distance':self.distList, 'target':self.y_train}

    def get_deftDist(self, tp):
        '''计算测试数据与预先序列的距离'''
        list = []  # 存放预测数据与预先序列的距离，列表含有k个距离
        '''
        for i in range(len(self.x_new)):
            tp = self.x_new[i]
        '''
        for j in range(len(self.idList)):
            sum = 0
            index = self.idList[j]
            for k in range(len(tp)):
                sum += (tp[k] - self.x_train[index][k]) ** 2
            sum = math.sqrt(sum)
            list.append(sum)
        return list

    def score(self, y_pre, y_test):
        '''计算精确值'''
        count = 0
        scoreList = list(map(lambda x: x[0] - x[1], zip(y_pre, y_test)))
        for i in scoreList:
            if i == 0:
                count += 1
        score = count / len(scoreList)
        return score

data = pd.read_excel(r"C:\Users\86153\Desktop\第3篇文章.xlsx", header=0)
label_need=data.keys()
X=data[label_need].values[:,1:5]
y=data[label_need].values[:,5]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
num_test = X_test.shape[0]

knn = MyKNN(n_neighbors=2)

knn.get_trainData(X_train, y_train)
y_pre,dists = knn.predict(X_test)

print('the kind of X_test:\n{}'.format(y_pre))  # 输出测试数据的预测类别
print('the score:{:.10}'.format(knn.score(y_pre, y_test)))  # 输出模型精准度
print('distances list : ', dists)  # 输出距离二维矩阵
print("KNN回归的误差MSE：", np.sum((y_pre - y_test)**2) /num_test)#输出 MSE 值
print("残差平方和SSR：",np.sum((y_pre - y_test)**2))# 残差平方和
print("总平方和SST：",np.sum((y_test - np.mean(y_test))**2))#总平方和
print("相关性R2:",1 - np.sum((y_pre - y_test)**2)/ np.sum((y_test - np.mean(y_test))**2))# R2指标
#print("KNN回归的R值:", np.square(np.sum(y_pre-np.mean(y_pre)*(y_test-np.mean(y_test)))/((np.sum(y_pre-np.mean(y_pre)))*np.sum(y_test-np.mean(y_test)))))#输出的 R 值

#print(np.sum((y_test-y_pre)))
#print(np.sum((y_test - y_pre)**2)/num_test)
#print("KNN回归的误差:", np.mean(np.sum((y_test - y_pre)**2)))#输出误差

'''结果可视化1'''
import matplotlib.pyplot as plt

xx = range(0, len(y_test))
plt.figure(figsize=(12, 6))
plt.scatter(xx, y_test, color="red", label="Sample Point", linewidth=3)
plt.plot(xx, y_pre, color="blue", label="Fitting Line", linewidth=2)
plt.legend()
plt.show()

'''结果可视化2'''
t = np.arange(len(y_test))
plt.figure(figsize=(18, 6))
plt.plot(t, y_test, "rs-", linewidth=1, label='Test(label)')
plt.plot(t, y_pre, 'go-', linewidth=1, label='Predict')
plt.legend()
plt.show()