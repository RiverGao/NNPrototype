# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:44:12 2019

@author: SAMSUNG
"""
'''This is an attempt on neuron network'''

import numpy as np
import random


def sigmoid(z):
    return 1.0/(1.0+ np.exp(-z))

def sigmoid_prime(z): # sigmoid 函数的特性决定的导数形式
    return sigmoid(z)*(1-sigmoid(z))


class Network(object):
    def __init__(self, sizes): #sizes为各层神经元数量的序列
        self.num_layers=len(sizes) #层数
        self.sizes=sizes #各层神经元个数
        self.biases= [np.random.randn(y,1) for y in sizes[1:]] #为输入层以后的各层，按神经元个数分配随机偏置
        # b[l][k]表示第 l+1 层第 k 个神经元的偏置，b[l] 是一个向量（纵向）
        self.weights= [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])] #为输出层以前的各层，按后一层神经元个数分配随机权重
        # w[l][k][j] 代表第 l 层的第 j 个神经元指向第 l+1 层的第 k 个神经元的权重
        # w[l][k] 表示第 l+1 层的第 k 个神经元的输入权重矩阵，w[l][k]·a[l]+b[l][k]==z[l+1][k]
        # w[l][k] 是一个 1 行的矩阵（横向）
    def feedfoward(self,a): #输入为 a, 求最终输出（用于评价精度）
        for b,w in zip(self.biases, self.weights):
            a=sigmoid(np.dot(w, a)+b)
        return a
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None): #随机梯度下降，输入训练集（二元组列表）、小数据集大小（整数），学习率，测试集        
        n=len(training_data) #用于采样
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] #打乱后取每 mini_batch_size 个数据对
            for mini_batch in mini_batches: #完成全部 batch 为一个 epoch
                self.update_mini_batch(mini_batch, eta) # 逐 batch 更新参数 w, b。update 算法在后面给出，需要用到反向传播
            if test_data!=None:
                n_test=len(test_data) #test_data 为可选参数，若输入则测试精度，否则直接跳过
                print('Epoch {0}: {1}/{2}'.format(j, self.evaluate(test_data), n_test)) # evaluate 方法在后面给出
            else:
                print('Epoch {0} completed!'.format(j))
            
    def update_mini_batch(self, mini_batch, eta):
        nabla_b=[np.zeros(b.shape) for b in self.biases] #用 0 初始化 C 关于每个 b 的偏导，从第二层开始排列
        nabla_w=[np.zeros(w.shape) for w in self.weights] #用 0 初始化 C 关于每个 w 的偏导，从第二层开始排列
        for x,y in mini_batch: #对每一对训练数据计算 C(x,y) 关于 w,b 的梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) #反向传播算法在后面给出
            nabla_b=[nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #更新梯度的 b 分量
            nabla_w=[nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #更新梯度的 w 分量
            self.weights=[w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
            self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)] #用更新的梯度更新权重和偏置矩阵

    def backprop(self, x, y):
        """返回一个元组 (nabla_b, nabla_w)，表示 C 关于所有 b, w 的梯度"""
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        #第一步：feedforward
        activation=x #第一个激活值，即输入的 x（向量）
        activations=[x] #用于逐层储存所有激活值的列表，从输入层开始
        zs=[] #用于逐层储存所有带权输入 z 的列表，从第二层开始
        for b,w in zip(self.biases, self.weights):
            z=np.dot(w, activation)+b #求出下一层的带权输入向量 z
            zs.append(z) #存储下一层的 z
            activation=sigmoid(z) #求出下一层的激活值向量
            activations.append(activation) #存储下一层的激活值向量
        #第二步：backpropogation
        delta=self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
            #计算输出层误差 delta(L)，其中 cost_derivative 根据代价函数选取有所不同，在后面给出
            # “ * ” 表示 Hadamard 乘
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta, activations[-2].transpose()) #此处转置操作存疑
        for l in range(2, self.num_layers): #此处 l 表示从后往前数
            z=zs[-l]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].transpose(), delta) * sp #求出这一层的误差，此处的转置是必要的
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta, activations[-l-1].transpose()) #此处转置存疑
        return (nabla_b, nabla_w)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y) #这是对于一对 (x,y) 的二次损失函数的导数
        
        
    def evaluate(self, test_data):
        test_results=[(np.argmax(self.feedfoward(x)), y) for (x,y) in test_data]
        #求最大值的操作表示找出输出值最大的神经元，仅适用于识别数字
        return sum(int(x==y) for (x,y) in test_results)
            
        











































        