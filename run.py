#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/8 2:42 下午
# @Author  : huangscar
# @Site    : 
# @File    : run.py
# @Software: PyCharm

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
import math
import logging
import traceback
import os
import random
import math
import pdb
Tensor = FloatTensor

# 参数
BATCH_SIZE = 2
LR = 0.001
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 340
Q_NETWORK_ITERATION = 100


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

logging.basicConfig(filename='my5.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

# 层
class MyLayer(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super(MyLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if torch.cuda.is_available():
            self.w_1 = nn.Parameter(torch.Tensor(1, self.in_features).cuda())
            self.w_2 = nn.Parameter(torch.Tensor(self.in_features, self.in_features).cuda())
            self.w_3 = nn.Parameter(torch.Tensor(self.in_features, self.in_features).cuda())
        else:
            self.w_1 = nn.Parameter(torch.Tensor(1, self.in_features))
            self.w_2 = nn.Parameter(torch.Tensor(self.in_features, self.in_features))
            self.w_3 = nn.Parameter(torch.Tensor(self.in_features, self.in_features))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(self.out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
        self.reset_parameters()

    def forward(self, x, d, s, batch_num, action_num):
        # print(s)
        # 转置，将行向量变成列向量
        x_t = x.permute(0, 2, 1)
        d_t = d.permute(0, 2, 1)
        s_t = s.permute(0, 2, 1)

        # 第一项：w_1*v(a)
        result_1 = self.w_1.matmul(x_t).squeeze()
        if batch_num == 1:
            result_1 = result_1.unsqueeze(0)

        # 第二项：v(a)_t*w_2*d
        result_2 = x.matmul(self.w_2).matmul(d_t)

        # 第三项：v(a)_t*w_3*s
        result_3 = x.matmul(self.w_3).matmul(s_t)

        # 取对角线
        if torch.cuda.is_available():
            mask = torch.arange(start=0, end=action_num, step=1).unsqueeze(0).unsqueeze(0).cuda()
        else:
            mask = torch.arange(start=0, end=action_num, step=1).unsqueeze(0).unsqueeze(0)
        mask = mask.repeat((batch_num, 1, 1))
        result_2 = result_2.gather(1, mask).squeeze()
        result_3 = result_3.gather(1, mask).squeeze()

        # 将b扩展成相应维度
        num_1 = list(result_1.size())[0]
        num_2 = list(result_1.size())[1]
        b = self.bias.repeat(num_1, num_2)

        return result_1 + result_2 + result_3 + b

    def reset_parameters(self):
        # stdv_1 = 1. / math.sqrt(self.w_1.size(0))
        self.w_1.data.normal_(-0.01, 0.01)
        # stdv_2 = 1. / math.sqrt(self.w_2.size(0))
        self.w_2.data.normal_(-0.01, 0.01)
        # stdv_3 = 1. / math.sqrt(self.w_3.size(0))
        self.w_3.data.normal_(-0.01, 0.01)

# 模型
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(4, 40, False)
        self.layer1.weight.data.normal_(-0.001, 0.001)
        # self.layer1.bias.data.normal_(-0.1, 0.1)
        self.layer2 = nn.Linear(8272, 1024)
        self.layer3 = nn.Linear(1024, 1, False)
        self.layer4 = MyLayer(4136, 1)

    # 输入变量说明：
    # all_box：     所有的 box 的特征向量矩阵                              维度为：[batch_num, 64, 64, 4096]  分别为batch_num，最大action_num，最大box_num，特征向量维度
    # all_box_pos： 所有 box 的坐标矩阵                                   维度为：[batch_num, 64, 64, 4]     分别为batch_num，最大action_num，最大box_num，坐标维度
    # chosen_box：  所有已经选择的 box 特征向量矩阵                         维度为：[batch_num, 64, 33, 4096]  分别为batch_num，最大action_num，最大chosen_num，特征向量维度
    # chosen_pos：  所有已经选择的 box 坐标矩阵                            维度为：[batch_num, 64, 33, 4]      分别为batch_num，最大action_num，最大chosen_num，坐标维度
    # action_box：  所有可选择的 box 特征向量矩阵                           维度为：[batch_num, 64, 4096]      分别为batch_num，最大action_num，特征向量维度
    # action_pos：  所有可选择的 box 坐标矩阵                              维度为：[batch_num, 64, 4]         分别为batch_num，最大action_num，坐标维度
    # action_a_box：用于和 all_box 拼接的可选择box的特征向量矩阵             维度为：[batch_num, 64, 64, 4096]  分别为batch_num，最大action_num，最大box_num，特征向量维度
    # action_a_pos：用于和 all_box 拼接的可选择的box坐标矩阵                维度为：[batch_num, 64, 64, 4]     分别为batch_num，最大action_num，最大box_num，坐标维度
    # action_c_box：用于和 chosen_box 拼接的可选择的box的特征向量矩阵        维度为：[batch_num, 64, 33, 4096]  分别为batch_num，最大action_num，特征向量维度
    # action_c_pos：用于和 chosen_box 拼接的可选择的box的坐标矩阵           维度为：[batch_num, 64, 33, 4]     分别为batch_num，最大action_num，坐标维度
    # box_num：     存储每个传入的 all_box 真实的 box 数量的 list
    # action_num：  最大可选择 box 的数量，这里均为 64
    # chosen_num：  每一条 batch 的已选择box的数量，list
    # batch_num：   batch 的数量

    def forward(self, all_box, all_box_pos, chosen_box, chosen_pos, action_box, action_pos, action_a_box, action_a_pos, action_c_box, action_c_pos, box_num, action_num, chosen_num, batch_num):

        if batch_num > 1:
            batch_num = list(all_box.shape)[0]

        # 获得坐标映射
        all_pos_map = self.layer1(all_box_pos)
        chosen_pos_map = self.layer1(chosen_pos)
        action_pos_map = self.layer1(action_pos)
        action_a_pos_map = self.layer1(action_a_pos)
        action_c_pos_map = self.layer1(action_c_pos)

        # 将坐标映射和特征向量矩阵拼接
        all_box = torch.cat((all_box, all_pos_map), dim=-1)
        chosen_box = torch.cat((chosen_box, chosen_pos_map), dim=-1)
        action_box = torch.cat((action_box, action_pos_map), dim=-1)
        action_a_box = torch.cat((action_a_box, action_a_pos_map), dim=-1)
        action_c_box = torch.cat((action_c_box, action_c_pos_map), dim=-1)

        # 获得chosen_box和all_box的w
        for i in range(batch_num):
            all_box_a = torch.cat((all_box[i], action_a_box[i]), dim=-1)
            chosen_box_a = torch.cat((chosen_box[i], action_c_box[i]), dim=-1)
            d_i = torch.tanh(self.layer2(all_box_a))
            d_i = self.layer3(d_i)
            s_i = torch.tanh(self.layer2(chosen_box_a))
            s_i = self.layer3(s_i)
            if i == 0:
                d = d_i.unsqueeze(0)
                s = s_i.unsqueeze(0)
            else:
                d = torch.cat((d, d_i.unsqueeze(0)), 0)
                s = torch.cat((s, s_i.unsqueeze(0)), 0)
        d = torch.exp(d)

        # 获得z
        if torch.cuda.is_available():
            z = torch.zeros((batch_num, action_num, 1, 1)).cuda()
        else:
            z = torch.zeros((batch_num, action_num, 1, 1))# batch_num, max_action_num, 1, 1
        for i in range(batch_num):
            for j in range(action_num):
                z[i, j, 0, 0] = torch.sum(d[i, j, :box_num[i], 0])
        d = torch.div(d, z)

        # chosen_box同理
        for i in range(batch_num):
            e = torch.bmm(d[i, :, :, :].permute(0, 2, 1), all_box[i, :, :, :])
            if i == 0:
                f = e.squeeze().unsqueeze(0)
            else:
                f = torch.cat((f, e.squeeze().unsqueeze(0)), 0)

        s = torch.exp(s)
        if torch.cuda.is_available():
            z = torch.zeros((batch_num, action_num, 1, 1)).cuda()
        else:
            z = torch.zeros((batch_num, action_num, 1, 1))
        for i in range(batch_num):
            for j in range(action_num):
                if chosen_num[i] != 0:
                    z[i, j, 0, 0] = torch.sum(s[i, j, :chosen_num[i], 0])
                else:
                    z[i, j, 0, 0] = torch.ones(1, 1, requires_grad=True).cuda()
        s = torch.div(s, z)
        s = torch.mul(s, chosen_box)
        s = torch.sum(s, -2, False)

        # action_box对应v(a)，f对应v(B_f, a), s对应v(S, a)
        output = self.layer4(action_box, f, s, batch_num, action_num)
        return output

# DQN
class DQN:
    def __init__(self):
        super(DQN, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.eval_net, self.target_net = QNetwork(), QNetwork()
        # self.eval_net = torch.load('eval_net2.pkl')
        # self.target_net = torch.load('target_net2.pkl')
        self.eval_net = QNetwork()
        self.target_net = QNetwork()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.eval_net = nn.DataParallel(self.eval_net)
            self.target_net = nn.DataParallel(self.target_net)
        self.eval_net.to(device)
        self.target_net.to(device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.box_num = 1

        # self.store_num = 0
        # self.all_box_arr = []
        self.all_box_list = []

        self.memory = np.zeros((MEMORY_CAPACITY, 64 * 2 + 3))

        self.loss_arr = []
        # self.grad_arr = []
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.SmoothL1Loss()

    # 存储all_box数据
    def set_data(self, all_box, pic_id):

        # 将所有图片的所有box记录在self.all_box_list的list中，其中index是pic_id
        num = len(self.all_box_list)
        # print(num)
        if pic_id > num-1:
            for i in range(pic_id - num + 1):
                self.all_box_list.append(0)
        if type(self.all_box_list[pic_id]) == int:
            self.all_box_list[pic_id] = all_box.copy()
        # self.c = np.zeros((1, self.box_num))

    # 计算 Q 值函数
    def get_q_utils(self, all_box_in, state, batch_num, box_num, net_type):
        chosen_num_list = []
        for i in range(batch_num):

            # 获得 action_box 和 chosen_box 下标，用于从 all_box 中筛选
            action_index = np.argwhere(state[i, :box_num[i]] == 0)[:, 0]
            chosen_index = np.argwhere(state[i, :box_num[i]] == 1)[:, 0]

            # 获得 action_box 和 chosen_box 的数量
            action_num = np.shape(action_index)[0]
            chosen_num = np.shape(chosen_index)[0]
            chosen_num_list.append(chosen_num)

            # 获得 action_box 矩阵，包括步骤获取 box，不够 64 的填充到64
            action_box_np = all_box_in[i:i+1, action_index, 0:4096]
            action_pos_np = all_box_in[i:i + 1, action_index, 4096:]
            action_box_np = np.pad(action_box_np, ((0, 0), (0, 64 - action_num), (0, 0)), 'constant')
            action_pos_np = np.pad(action_pos_np, ((0, 0), (0, 64 - action_num), (0, 0)), 'constant')

            # 获得 all_box 矩阵，包括步骤获取，重复 action_num 次（用于和action_box拼接），如果 action 不够 64 次填充到 64
            all_box_np = all_box_in[i:i+1, :, 0:4096]
            all_pos_np = all_box_in[i:i+1, :, 4096:]
            all_box_np_three_a = np.repeat(all_box_np, action_num, axis=0)
            all_pos_np_three_a = np.repeat(all_pos_np, action_num, axis=0)
            all_box_np_three_a = np.pad(all_box_np_three_a, ((0, 64 - action_num), (0, 0), (0, 0)), 'constant')
            all_pos_np_three_a = np.pad(all_pos_np_three_a, ((0, 64 - action_num), (0, 0), (0, 0)), 'constant')

            # 获得 chosen_box 矩阵，包括步骤获取，不足33次填充到33，其余部分和上面相同
            chosen_box_np = all_box_in[i:i+1, chosen_index, 0:4096]
            chosen_pos_np = all_box_in[i:i+1, chosen_index, 4096:]
            chosen_box_np = np.pad(chosen_box_np, ((0, 0), (0, 34 - chosen_num), (0, 0)), 'constant')
            chosen_pos_np = np.pad(chosen_pos_np, ((0, 0), (0, 34 - chosen_num), (0, 0)), 'constant')
            chosen_box_np_three_a = np.repeat(chosen_box_np, action_num, axis=0)  # action_num
            chosen_pos_np_three_a = np.repeat(chosen_pos_np, action_num, axis=0)  # action_num
            chosen_box_np_three_a = np.pad(chosen_box_np_three_a, ((0, 64 - action_num), (0, 0), (0, 0)), 'constant')
            chosen_pos_np_three_a = np.pad(chosen_pos_np_three_a, ((0, 64 - action_num), (0, 0), (0, 0)), 'constant')

            # 获得 action_a_box 矩阵，包括步骤获取，不足64次填充到64，升维，每一个action_box重复box_num次，box_num不足64填充到64
            action_a_box_np = all_box_in[i, action_index, 0:4096]
            action_a_pos_np = all_box_in[i, action_index, 4096:]
            action_a_box_np = np.pad(action_a_box_np, ((0, 64 - action_num), (0, 0)), 'constant')
            action_a_pos_np = np.pad(action_a_pos_np, ((0, 64 - action_num), (0, 0)), 'constant')
            action_a_box_np_three = action_a_box_np[:, np.newaxis, :]
            action_a_pos_np_three = action_a_pos_np[:, np.newaxis, :]
            action_a_box_np_three_a = np.repeat(action_a_box_np_three, box_num[i], axis=1)  # all_box_num
            action_a_pos_np_three_a = np.repeat(action_a_pos_np_three, box_num[i], axis=1)  # all_box_num
            action_a_box_np_three_a = np.pad(action_a_box_np_three_a, ((0, 0), (0, 64-box_num[i]), (0, 0)),
                                             'constant')  # (0, 0), (0, max_box_num - box_nun), (0, 0)
            action_a_pos_np_three_a = np.pad(action_a_pos_np_three_a, ((0, 0), (0, 64 - box_num[i]), (0, 0)),
                                             'constant')  # (0, 0), (0, max_box_num - box_nun), (0, 0)

            # 获得 action_c_box 矩阵，包括步骤获取，不足64次填充到64，升维，每一个action_box重复chosen_num次，chosen_num不足64填充到64
            action_c_box_np = all_box_in[i, action_index, 0:4096]
            action_c_pos_np = all_box_in[i, action_index, 4096:]
            action_c_box_np = np.pad(action_c_box_np, ((0, 64 - action_num), (0, 0)), 'constant')
            action_c_pos_np = np.pad(action_c_pos_np, ((0, 64 - action_num), (0, 0)), 'constant')
            action_c_box_np_three = action_c_box_np[:, np.newaxis, :]
            action_c_pos_np_three = action_c_pos_np[:, np.newaxis, :]
            action_c_box_np_three_a = np.repeat(action_c_box_np_three, chosen_num, axis=1)  # all_box_num
            action_c_pos_np_three_a = np.repeat(action_c_pos_np_three, chosen_num, axis=1)  # all_box_num
            action_c_box_np_three_a = np.pad(action_c_box_np_three_a, ((0, 0), (0, 34 - chosen_num), (0, 0)),
                                             'constant')  # (0, 0), (0, max_box_num - box_nun), (0, 0)
            action_c_pos_np_three_a = np.pad(action_c_pos_np_three_a, ((0, 0), (0, 34 - chosen_num), (0, 0)),
                                             'constant')  # (0, 0), (0, max_box_num - box_nun), (0, 0)

            if i == 0:
                action_box_np_three_b = action_box_np
                action_pos_np_three_b = action_pos_np
                chosen_box_np_three_b = chosen_box_np_three_a[np.newaxis, :, :, :]
                chosen_pos_np_three_b = chosen_pos_np_three_a[np.newaxis, :, :, :]
                all_box_np_three_b = all_box_np_three_a[np.newaxis, :, :, :]
                all_pos_np_three_b = all_pos_np_three_a[np.newaxis, :, :, :]
                action_a_box_np_three_b = action_a_box_np_three_a[np.newaxis, :, :, :]
                action_a_pos_np_three_b = action_a_pos_np_three_a[np.newaxis, :, :, :]
                action_c_box_np_three_b = action_c_box_np_three_a[np.newaxis, :, :, :]
                action_c_pos_np_three_b = action_c_pos_np_three_a[np.newaxis, :, :, :]
            else:
                action_box_np_three_b = np.vstack((action_box_np_three_b, action_box_np))
                action_pos_np_three_b = np.vstack((action_pos_np_three_b, action_pos_np))
                chosen_box_np_three_b = np.vstack((chosen_box_np_three_b, chosen_box_np_three_a[np.newaxis, :, :, :]))
                chosen_pos_np_three_b = np.vstack((chosen_pos_np_three_b, chosen_pos_np_three_a[np.newaxis, :, :, :]))
                all_box_np_three_b = np.vstack((all_box_np_three_b, all_box_np_three_a[np.newaxis, :, :, :]))
                all_pos_np_three_b = np.vstack((all_pos_np_three_b, all_pos_np_three_a[np.newaxis, :, :, :]))
                action_a_box_np_three_b = np.vstack((action_a_box_np_three_b, action_a_box_np_three_a[np.newaxis, :, :, :]))
                action_a_pos_np_three_b = np.vstack((action_a_pos_np_three_b, action_a_pos_np_three_a[np.newaxis, :, :, :]))
                action_c_box_np_three_b = np.vstack(
                    (action_c_box_np_three_b, action_c_box_np_three_a[np.newaxis, :, :, :]))
                action_c_pos_np_three_b = np.vstack(
                    (action_c_pos_np_three_b, action_c_pos_np_three_a[np.newaxis, :, :, :]))

        # 生成tensor
        if torch.cuda.is_available():
            all_box = torch.FloatTensor(all_box_np_three_b).cuda()
            all_box.requires_grad_()
            all_pos = torch.FloatTensor(all_pos_np_three_b).cuda()
            all_pos.requires_grad_()
            chosen_box = torch.FloatTensor(chosen_box_np_three_b).cuda()
            chosen_box.requires_grad_()
            chosen_pos = torch.FloatTensor(chosen_pos_np_three_b).cuda()
            chosen_pos.requires_grad_()
            action_box = torch.FloatTensor(action_box_np_three_b).cuda()
            action_box.requires_grad_()
            action_pos = torch.FloatTensor(action_pos_np_three_b).cuda()
            action_pos.requires_grad_()
            action_a_box = torch.FloatTensor(action_a_box_np_three_b).cuda()
            action_a_box.requires_grad_()
            action_a_pos = torch.FloatTensor(action_a_pos_np_three_b).cuda()
            action_a_pos.requires_grad_()
            action_c_box = torch.FloatTensor(action_c_box_np_three_b).cuda()
            action_c_box.requires_grad_()
            action_c_pos = torch.FloatTensor(action_c_pos_np_three_b).cuda()
            action_c_pos.requires_grad_()
        else:
            all_box = torch.FloatTensor(all_box_np_three_b)
            all_box.requires_grad_()
            all_pos = torch.FloatTensor(all_pos_np_three_b)
            all_pos.requires_grad_()
            chosen_box = torch.FloatTensor(chosen_box_np_three_b)
            chosen_box.requires_grad_()
            chosen_pos = torch.FloatTensor(chosen_pos_np_three_b)
            chosen_pos.requires_grad_()
            action_box = torch.FloatTensor(action_box_np_three_b)
            action_box.requires_grad_()
            action_pos = torch.FloatTensor(action_pos_np_three_b)
            action_pos.requires_grad_()
            action_a_box = torch.FloatTensor(action_a_box_np_three_b)
            action_a_box.requires_grad_()
            action_a_pos = torch.FloatTensor(action_a_pos_np_three_b)
            action_a_pos.requires_grad_()
            action_c_box = torch.FloatTensor(action_c_box_np_three_b)
            action_c_box.requires_grad_()
            action_c_pos = torch.FloatTensor(action_c_pos_np_three_b)
            action_c_pos.requires_grad_()

        # 代入网络计算
        if net_type == 1:
            q = self.eval_net(all_box, all_pos, chosen_box, chosen_pos, action_box, action_pos, action_a_box, action_a_pos,
                     action_c_box, action_c_pos, box_num, 64, chosen_num_list, batch_num)
        else:
            q = self.target_net(all_box, all_pos, chosen_box, chosen_pos, action_box, action_pos, action_a_box, action_a_pos,
                     action_c_box, action_c_pos, box_num, 64, chosen_num_list, batch_num)

        return q

    def choose_action(self, c, index):
        eps = 0.05 + (EPISILO - 0.05) * math.exp(-1. * self.learn_step_counter / 200)
        num = len(index)
        # print(num)
        if np.random.randn() <= eps:  # greedy policy
            box_num = []  # 记录box的数量

            # 获得产生序列的图片的所有box
            all_box = np.zeros((num, 64, 4100))
            for i in range(num):
                box = self.all_box_list[index[i]]
                box_num_i = np.shape(box)[0]
                all_box_i = np.pad(box, ((0, 64 - box_num_i), (0, 0)), 'constant')
                all_box_i = all_box_i[np.newaxis, :, :]
                box_num.append(box_num_i)
                all_box[i, :, :] = all_box_i

            # 获得产生序列的图片的q值
            q = self.get_q_utils(all_box, c, num, box_num, 1)

            # 记录添加的box所在的index（对于所有box的index）
            action = np.zeros((num, 1), dtype=int)
            for i in range(num):

                # 获得未选择的box对于所有的box的index
                action_index = np.argwhere(c[i, :box_num[i]] == 0)[:, 0]
                action_num = np.shape(action_index)[0]
                # 获得使得q值最大的box对于所有未选择的box的index
                max_index = q[i, :action_num].argmax(dim=0)
                action[i, 0] = action_index[max_index]

        else:  # random policy
            action = np.zeros((num, 1), dtype=int)
            for i in range(num):
                box = self.all_box_list[index[i]]
                box_num_i = np.shape(box)[0]
                while True:
                    action_i = np.random.randint(0, box_num_i)
                    if c[i, action_i] == 0:
                        break
                action[i, 0] = action_i
        return action

    # 用于测试步骤计算q值的函数，目前未使用
    def choose_action_test(self, c):
        all_box = np.pad(self.all_box, ((0, 64 - self.box_num), (0, 0)), 'constant')
        all_box = all_box[np.newaxis, :, :]
        box_num = []
        box_num.append(self.box_num)
        q = self.get_q_utils(all_box, c, 1, box_num, 1)
        action_index = np.argwhere(c == 0)[:, 1]
        action_num = np.shape(action_index)[0]
        max_index = q[0, :action_num].argmax(dim=0)
        action = action_index[max_index]
        return action

    # 存储有效记录，无效记录将不会被存储
    def store_transition(self, state, action, reward, next_state, pic_id):

        # 有效记录：本次选择的box是正确的box
        # 获得有效记录的index
        index = np.argwhere(reward < 0)[:, 0]

        # 筛选有效记录
        state = state[index, :]
        action = action[index, :]
        reward = reward[index, :]
        next_state = next_state[index, :]
        pic_id = pic_id[index, :]

        transition = np.hstack((pic_id, state, action, reward, next_state))

        num = np.shape(transition)[0]
        if num != 0:
            # print(transition)
            index = self.memory_counter % MEMORY_CAPACITY
            self.memory[index:index + num, :] = transition
            self.memory_counter += num

    def learn(self):

        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # print(self.memory)
        if self.memory_counter > BATCH_SIZE:

            # 当存储的memory未满，防止选择错误的记录
            if self.memory_counter < MEMORY_CAPACITY:
                memory_num = self.memory_counter
            else:
                memory_num = MEMORY_CAPACITY

            # 获取抽取的记录的各个矩阵
            sample_index = np.random.choice(memory_num, BATCH_SIZE)
            batch_memory = self.memory[sample_index, :]
            batch_pic_id = batch_memory[:, 0:1]
            batch_state = batch_memory[:, 1:65]
            batch_action = batch_memory[:, 65:66]
            batch_reward = batch_memory[:, 66:67]
            batch_next_state = batch_memory[:, -64:]
            self.learn_step_counter += 1

            box_num_list = []  # 记录抽取的记录所在的图片的box的数量

            # 获得抽取的记录所在的图片all_box
            index = np.zeros((BATCH_SIZE, 2), dtype=int)  # 记录抽取的记录选取的box对于未选取的box的index
            for i in range(BATCH_SIZE):
                all_box_np = self.all_box_list[int(batch_pic_id[i, 0])]
                box_num = np.shape(all_box_np)[0]
                box_num_list.append(box_num)
                all_box_np = np.pad(all_box_np, ((0, 64 - box_num), (0, 0)), 'constant')
                all_box_np = all_box_np[np.newaxis, :, :]

                # print(np.argwhere(batch_state[i] == 0))
                action_index = np.argwhere(batch_state[i] == 0)[:, 0]
                index[i, 0] = i
                index[i, 1] = np.argwhere(action_index == batch_action[i, 0])

                if i == 0:
                    all_box = all_box_np
                else:
                    all_box = np.vstack((all_box, all_box_np))

            q_eval = self.get_q_utils(all_box, batch_state, BATCH_SIZE, box_num_list, 1)
            # logging.info('finish_eval')

            # 获得抽取的记录（匹配选择的Box）的q值
            q_eval_true = q_eval[index[:, 0], index[:, 1]]

            # 获得抽取的记录的下一步的q值
            q_next = self.get_q_utils(all_box, batch_next_state, BATCH_SIZE, box_num_list, 2)

            # 获取抽取的记录的下一步的最大q值
            for i in range(BATCH_SIZE):

                # 获得抽取记录的下一个状态中未选择box的真实数量
                action_num = np.shape(np.argwhere(batch_next_state[i] == 0)[:, 0])[0] - (64 - box_num_list[i])
                # 获得抽取记录的下一个状态的真实最大q值
                q_next_max_i = q_next[i, :action_num].max(0)[0].unsqueeze(0)
                if i == 0:
                    q_next_max = q_next_max_i
                else:
                    q_next_max = torch.cat((q_next_max, q_next_max_i), 0)

            # 将reward转成tensor
            if torch.cuda.is_available():
                batch_reward_ten = torch.FloatTensor(batch_reward).cuda()
            else:
                batch_reward_ten = torch.FloatTensor(batch_reward)

            # 获得q_target
            q_target = batch_reward_ten.t().squeeze() + GAMMA * q_next_max

            # 计算loss及梯度下降
            loss = self.loss_func(q_eval_true, q_target)
            self.loss_arr.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            # params = list(self.eval_net.named_parameters())
            # # print(params)
            # for (name, param) in params:
            #     print('name: ', name)
            #     print('grad: ', param.grad)
            self.optimizer.step()

    def get_q(self, state, action):
        all_box = np.pad(self.all_box, ((0, 64 - self.box_num), (0, 0)), 'constant')
        all_box = all_box[np.newaxis, :, :]
        box_num = []
        box_num.append(self.box_num)
        q = self.get_q_utils(all_box, state, 1, box_num, 1)
        action_index = np.argwhere(state == 0)[:, 1]
        index = np.argwhere(action_index == action)[:, 0][0]
        return q[0, index]



    def print_loss(self):
        # print(self.loss_arr)
        logging.info(str(self.loss_arr))
        finish = True
        for loss in self.loss_arr:
            if torch.cuda.is_available():
                if loss.cpu().detach().numpy() > 0.01:
                    finish = False
            else:
                # print(loss)
                # print(loss.detach())
                # print(loss.detach().numpy())
                if loss.detach().numpy() > 0.01:
                    finish = False
        self.loss_arr.clear()
        return finish

    def save_model(self):
        torch.save(self.eval_net, 'eval_net5.pkl')
        torch.save(self.target_net, 'target_net5.pkl')


def get_score(c, c_true):
    up = 0.0
    down = 0.0
    num = len(c_true)
    score = np.zeros((num, 1))
    # print('c_true: ', c_true)
    for j in range(num):
        for i in range(len(c_true[j])):
            if c[j, i] == c_true[j][i] and c[j, i] == 1:
                up += 1.0
                down += 1.0
            elif c_true[j][i] == 1:
                down += 1.0
            elif c[j, i] == 1:
                down += 1.0
        if down == 0:
            score[j, 0] = 0
        else:
            logging.info("up: " + str(up))
            score[j, 0] = up/down
        up = 0.0
        down = 0.0
    return score


if __name__ == '__main__':

    try:
        root_dir = '/home/wby/wby/wby_outputs/train/sgdet_train/train_img_predinfo_txt_noconstraint'
        dqn = DQN()
        # dqn = torch.load('')
        file_list = os.listdir(root_dir)
        times = 0
        finish = False

        file_dir = '/home/wby/wby/wby_outputs/train/sgdet_train/train_guildo_minnum_noconstraint.txt'
        file = open(file_dir, mode='r')
        add_true_box = []
        for line in file:
            words = line.split(' ')
            score = float(words[2])
            if score == 0.0:
                box_id = int(words[1])
                add_true_box.append(box_id)

        # 训练
        while True:
            if times > 0:
                logging.info("================================")
            # rand_txt = random.choice(file_list)
            rand_txt = file_list[0]
            path = root_dir + '/' + rand_txt
            f = open(path, mode='r')
            last_pic = -1
            # last_pic = 0
            eig_vec_all = []
            c_true = []
            c_true_list = []
            eig_vec_list = []
            learn_pic_list = []
            step = 0
            true_num = 0
            pic_num = 0
            for line in f:
                words = line.split(' ')
                pic_id = int(words[0])
                box_id = int(words[1])
                box_reward = float(words[2])
                box_reward = int(box_reward)
                if box_reward == 0:
                    if box_id in add_true_box:
                        box_reward = 1
                label = int(words[3])
                eig_vec = []
                for i in range(5, 4101):
                    num = float(words[i])
                    eig_vec.append(num)
                for i in range(-6, -2):
                    x1 = float(words[i])
                    eig_vec.append(x1)

                if last_pic == -1:
                    last_pic = pic_id

                if last_pic != pic_id:
                    if true_num >= 11:
                        logging.info("true num: " + str(true_num))
                        pic_num += 1
                        num = len(eig_vec_all)
                        eig_mat = np.array(eig_vec_all)
                        # print(eig_mat)
                        dqn.set_data(eig_mat, last_pic)
                        eig_vec_list.append(eig_mat)
                        learn_pic_list.append(last_pic)
                        c_true_list.append(c_true.copy())

                    eig_vec_all.clear()
                    c_true.clear()
                    true_num = 0
                    # print(last_pic)
                    logging.info("pic: " + str(last_pic))
                    logging.info("---------------------------")
                    if pic_num >= 10:
                        break
                    eig_vec_all.append(eig_vec)
                    c_true.append(box_reward)
                    last_pic = pic_id
                    if box_reward == 1:
                        true_num += 1
                else:
                    # times += 1
                    if box_reward == 1:
                        true_num += 1
                    eig_vec_all.append(eig_vec)
                    c_true.append(box_reward)
            f.close()
            learn_num = len(learn_pic_list)
            get_list_times = math.ceil(learn_num/4)
            for j in range(get_list_times):
                num = 4
                if j == get_list_times - 1:
                    num = learn_num - j*4
                last_score = np.zeros((num, 1), dtype=float)
                c = np.zeros((num, 64), dtype=int)
                index = learn_pic_list[j*4: j*4+num]
                for i in range(35 - 1):
                    action = dqn.choose_action(c, index)
                    # pdb.set_trace()
                    # print('action: ', action)
                    # logging.info("aciton: " + str(action))
                    c_last = c.copy()
                    line_index = [0, 1, 2, 3]
                    line_index = line_index[:num]
                    c[line_index, action.transpose()] = 1
                    if i != 35 - 2:
                        # print(c)
                        # print(c_true)
                        score = get_score(c, c_true_list[j*4:j*4+num])
                        reward = last_score - score
                        last_score = score.copy()
                        # logging.info("reward: " + str(reward))
                        logging.info("score: " + str(score))
                    else:
                        score = get_score(c, c_true_list[j*4:j*4+num])
                        reward = score
                    pic_id = np.mat(index).transpose()
                    # print('c_last; ', c_last, 'action: ', action, 'reward: ', reward, 'c: ', c, 'pic_id: ', pic_id)
                    # pdb.set_trace()
                    dqn.store_transition(c_last, action, reward, c, pic_id)
                if times > 0:
                    dqn.learn()
                    finish = dqn.print_loss()
                    dqn.save_model()
                    if finish:
                        break
            times += 1
            if times >= 200:
                break
        dqn.save_model()

        # 测试代码
        # times = 0
        # for file in file_list:
        #     times += 1
        #     path = root_dir + '/' + file
        #     f = open(path, mode='r')
        #     last_pic = -1
        #     eig_vec_all = []
        #     c_true = []
        #     step = 0
        #     reward_all = []
        #     pic_num = 0
        #     true_num = 0
        #     for line in f:
        #         words = line.split(' ')
        #         pic_id = int(words[0])
        #         box_id = int(words[1])
        #         box_reward = float(words[2])
        #         box_reward = int(box_reward)
        #         label = int(words[3])
        #         eig_vec = []
        #         for i in range(5, 4101):
        #             num = float(words[i])
        #             eig_vec.append(num)
        #         for i in range(-6, -2):
        #             x1 = float(words[i])
        #             eig_vec.append(x1)
        #         if last_pic == -1:
        #             last_pic = pic_id
        #         if last_pic < pic_id:
        #             if true_num >= 1:
        #                 pic_num += 1
        #                 num = len(eig_vec_all)
        #                 c = np.zeros((1, num))
        #                 eig_mat = np.array(eig_vec_all)
        #                 dqn.set_data(eig_mat, last_pic)
        #                 last_score = 0
        #                 Q = np.zeros((1, 35), dtype=float)
        #                 c_all = np.zeros((34, num))
        #                 for i in range(35):
        #                     action = dqn.choose_action_test(c)
        #                     q = dqn.get_q(c, action)
        #                     c_last = c
        #                     c[0, action] = 1
        #                     c_all[i + 1:i + 2, :] = c
        #                     Q[0, i] = q
        #                 s_max = - sys.float_info.max
        #                 max_i = 0
        #                 print(Q)
        #                 for i in range(1, 35):
        #                     s_i = Q[0, i] + (1 - GAMMA) * (np.sum(Q[:, i + 1:-1], axis=1)) - GAMMA * Q[:, -1]
        #                     if s_i > s_max:
        #                         max_i = i
        #                         s_max = s_i
        #                         print(s_max)
        #
        #                 c_max = c_all[max_i:max_i + 1, :]
        #                 print(c_max)
        #                 print(c_true)
        #                 reward = get_score(c_max, c_true)
        #                 reward_all.append(reward)
        #                 print(reward)
        #                 logging.info("reward: " + str(reward))
        #             eig_vec_all.clear()
        #             c_true.clear()
        #             true_num = 0
        #             if box_reward == 1:
        #                 true_num += 1
        #             if pic_num >= 10:
        #                 break
        #             eig_vec_all.append(eig_vec)
        #             c_true.append(box_reward)
        #             last_pic = pic_id
        #
        #         else:
        #             if box_reward == 1:
        #                 true_num += 1
        #             eig_vec_all.append(eig_vec)
        #             c_true.append(box_reward)
        #     f.close()
    except Exception as e:
        log = logging.getLogger("log error")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        # 日志名称
        file_handler = logging.FileHandler("error.txt")
        log.setLevel("DEBUG")
        file_handler.setFormatter(fmt)
        log.addHandler(file_handler)
        # 将堆栈中的信息输入到log上
        log.debug(traceback.format_exc())

