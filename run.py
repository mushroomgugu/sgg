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
Tensor = FloatTensor

# 参数
BATCH_SIZE = 4
LR = 0.001
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 3400
Q_NETWORK_ITERATION = 100


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

logging.basicConfig(filename='my4.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

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
        x_t = x.permute(0, 2, 1)
        d_t = d.permute(0, 2, 1)
        s_t = s.permute(0, 2, 1)

        result_1 = self.w_1.matmul(x_t).squeeze()
        if batch_num == 1:
            result_1 = result_1.unsqueeze(0)
        result_2 = x.matmul(self.w_2).matmul(d_t)
        result_3 = x.matmul(self.w_3).matmul(s_t)
        if torch.cuda.is_available():
            mask = torch.arange(start=0, end=action_num, step=1).unsqueeze(0).unsqueeze(0).cuda()
        else:
            mask = torch.arange(start=0, end=action_num, step=1).unsqueeze(0).unsqueeze(0)
        mask = mask.repeat((batch_num, 1, 1))
        result_2 = result_2.gather(1, mask).squeeze()
        result_3 = result_3.gather(1, mask).squeeze()

        # print(list(result_1.size()))
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

    def forward(self, all_box, all_box_pos, chosen_box, chosen_pos, action_box, action_pos, action_a_box, action_a_pos, action_c_box, action_c_pos, box_num, action_num, chosen_num, batch_num):

        if batch_num > 1:
            batch_num = list(all_box.shape)[0]

        all_pos_map = self.layer1(all_box_pos)
        chosen_pos_map = self.layer1(chosen_pos)
        action_pos_map = self.layer1(action_pos)
        action_a_pos_map = self.layer1(action_a_pos)
        action_c_pos_map = self.layer1(action_c_pos)
        # print(chosen_pos)
        # print('chosen_pos_map\n')
        # print(chosen_pos_map)

        all_box = torch.cat((all_box, all_pos_map), dim=-1)
        chosen_box = torch.cat((chosen_box, chosen_pos_map), dim=-1)
        action_box = torch.cat((action_box, action_pos_map), dim=-1)
        action_a_box = torch.cat((action_a_box, action_a_pos_map), dim=-1)
        action_c_box = torch.cat((action_c_box, action_c_pos_map), dim=-1)

        # all_box_a = torch.cat((all_box, action_a_box), dim=-1)
        # chosen_box_a = torch.cat((chosen_box, action_c_box), dim=-1)
        # d = torch.tanh(self.layer2(all_box_a))
        # d = self.layer3(d)
        # s = torch.tanh(self.layer2(chosen_box_a))
        # s = self.layer3(s)

        for i in range(batch_num):
            # print(all_box[i])
            # print(action_a_box[i])
            all_box_a = torch.cat((all_box[i], action_a_box[i]), dim=-1)
            chosen_box_a = torch.cat((chosen_box[i], action_c_box[i]), dim=-1)
            # print(all_box_a)
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
        if torch.cuda.is_available():
            z = torch.zeros((batch_num, action_num, 1, 1)).cuda()
        else:
            z = torch.zeros((batch_num, action_num, 1, 1))# batch_num, max_action_num, 1, 1
        for i in range(batch_num):
            for j in range(action_num):
                z[i, j, 0, 0] = torch.sum(d[i, j, :box_num[i], 0])
        d = torch.div(d, z)
        # d = torch.mul(d, all_box)
        #
        # d = torch.sum(d, -2, False)

        for i in range(batch_num):
            # print(d[i, :, :, :].shape)
            # print(all_box[i, :, :, :].shape)
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
        # print(s)
        for i in range(batch_num):
            for j in range(action_num):
                if chosen_num[i] != 0:
                    z[i, j, 0, 0] = torch.sum(s[i, j, :chosen_num[i], 0])
                else:
                    z[i, j, 0, 0] = torch.ones(1, 1, requires_grad=True).cuda()
        s = torch.div(s, z)
        s = torch.mul(s, chosen_box)
        # print(chosen_box)
        s = torch.sum(s, -2, False)

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

    def set_data(self, all_box, pic_id):
        self.box_num = np.shape(all_box)[0]
        self.all_box = all_box
        num = len(self.all_box_list)
        # print(num)
        if pic_id > num-1:
            for i in range(pic_id - num + 1):
                self.all_box_list.append(0)
        if type(self.all_box_list[pic_id]) == int:
            self.all_box_list[pic_id] = all_box
        # self.c = np.zeros((1, self.box_num))

    def get_q_utils(self, all_box_in, state, batch_num, box_num, net_type):
        chosen_num_list = []
        for i in range(batch_num):
            action_index = np.argwhere(state[i, :box_num[i]] == 0)[:, 0]
            chosen_index = np.argwhere(state[i, :box_num[i]] == 1)[:, 0]

            action_num = np.shape(action_index)[0]
            chosen_num = np.shape(chosen_index)[0]
            chosen_num_list.append(chosen_num)

            action_box_np = all_box_in[i:i+1, action_index, 0:4096]
            action_pos_np = all_box_in[i:i + 1, action_index, 4096:]
            action_box_np = np.pad(action_box_np, ((0, 0), (0, 64 - action_num), (0, 0)), 'constant')
            action_pos_np = np.pad(action_pos_np, ((0, 0), (0, 64 - action_num), (0, 0)), 'constant')

            all_box_np = all_box_in[i:i+1, :, 0:4096]
            all_pos_np = all_box_in[i:i+1, :, 4096:]
            all_box_np_three_a = np.repeat(all_box_np, action_num, axis=0)
            all_pos_np_three_a = np.repeat(all_pos_np, action_num, axis=0)
            all_box_np_three_a = np.pad(all_box_np_three_a, ((0, 64 - action_num), (0, 0), (0, 0)), 'constant')
            all_pos_np_three_a = np.pad(all_pos_np_three_a, ((0, 64 - action_num), (0, 0), (0, 0)), 'constant')

            chosen_box_np = all_box_in[i:i+1, chosen_index, 0:4096]
            chosen_pos_np = all_box_in[i:i+1, chosen_index, 4096:]
            chosen_box_np = np.pad(chosen_box_np, ((0, 0), (0, 34 - chosen_num), (0, 0)), 'constant')
            chosen_pos_np = np.pad(chosen_pos_np, ((0, 0), (0, 34 - chosen_num), (0, 0)), 'constant')
            chosen_box_np_three_a = np.repeat(chosen_box_np, action_num, axis=0)  # action_num
            chosen_pos_np_three_a = np.repeat(chosen_pos_np, action_num, axis=0)  # action_num
            chosen_box_np_three_a = np.pad(chosen_box_np_three_a, ((0, 64 - action_num), (0, 0), (0, 0)), 'constant')
            chosen_pos_np_three_a = np.pad(chosen_pos_np_three_a, ((0, 64 - action_num), (0, 0), (0, 0)), 'constant')

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

        # if batch_num > 1:
        #     logging.info('start_ten')
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

        # if batch_num > 1:
        #     logging.info('start_net')

        if net_type == 1:
            q = self.eval_net(all_box, all_pos, chosen_box, chosen_pos, action_box, action_pos, action_a_box, action_a_pos,
                     action_c_box, action_c_pos, box_num, 64, chosen_num_list, batch_num)
        else:
            q = self.target_net(all_box, all_pos, chosen_box, chosen_pos, action_box, action_pos, action_a_box, action_a_pos,
                     action_c_box, action_c_pos, box_num, 64, chosen_num_list, batch_num)

        return q

    def choose_action(self, c):
        eps = 0.05 + (EPISILO - 0.05) * math.exp(-1. * self.learn_step_counter / 200)
        if np.random.randn() <= eps:  # greedy policy
            all_box = np.pad(self.all_box, ((0, 64 - self.box_num), (0, 0)), 'constant')
            all_box = all_box[np.newaxis, :, :]
            box_num = []
            box_num.append(self.box_num)
            q = self.get_q_utils(all_box, c, 1, box_num, 1)
            action_index = np.argwhere(c == 0)[:, 1]
            action_num = np.shape(action_index)[0]
            max_index = q[0, :action_num].argmax(dim=0)
            action = action_index[max_index]

        else:  # random policy
            while True:
                action = np.random.randint(0, self.box_num)
                if c[0, action] == 0:
                    break
        return action

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

    def store_transition(self, state, action, reward, next_state, pic_id):
        pic_id_r = np.zeros((1, 1))
        pic_id_r[0, 0] = pic_id
        a_r = np.zeros((1, 2))
        a_r[0, 0] = action
        a_r[0, 1] = reward
        transition = np.hstack((pic_id_r, state, a_r, next_state))
        # print(pic_id_r.shape)
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # print(self.memory)
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_pic_id = batch_memory[:, 0:1]
        batch_state = batch_memory[:, 1:65]
        batch_action = batch_memory[:, 65:66]
        batch_reward = batch_memory[:, 66:67]
        batch_next_state = batch_memory[:, -64:]
        self.learn_step_counter += 1

        box_num_list = []
        index = np.zeros((BATCH_SIZE, 2), dtype=int)
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

        # logging.info('q_eval')
        q_eval = self.get_q_utils(all_box, batch_state, BATCH_SIZE, box_num_list, 1)
        # logging.info('finish_eval')

        q_eval_true = q_eval[index[:, 0], index[:, 1]]

        q_next = self.get_q_utils(all_box, batch_next_state, BATCH_SIZE, box_num_list, 2)
        # print(q_next.shape)
        for i in range(BATCH_SIZE):
            # print(q_next[i, :box_num[i]].max(0))
            q_next_max_i = q_next[i, :box_num_list[i]].max(0)[0].unsqueeze(0)
            # print(q_next_max_i.shape)
            if i == 0:
                q_next_max = q_next_max_i
            else:
                q_next_max = torch.cat((q_next_max, q_next_max_i), 0)

        if torch.cuda.is_available():
            batch_reward_ten = torch.FloatTensor(batch_reward).cuda()
        else:
            batch_reward_ten = torch.FloatTensor(batch_reward)
        # print(batch_reward_ten.shape)
        q_target = batch_reward_ten.t().squeeze() + GAMMA*q_next_max

        # print(q_eval_true.shape)
        # print(q_target.shape)
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
        torch.save(self.eval_net, 'eval_net4.pkl')
        torch.save(self.target_net, 'target_net4.pkl')


def get_score(c, c_true):
    up = 0.0
    down = 0.0
    finish = True
    for i in range(len(c_true)):
        if c[0, i] == c_true[i] and c[0, i] == 1:
            up += 1.0
            down += 1.0
        elif c_true[i] == 1:
            down += 1.0
            finish = False
        elif c[0, i] == 1:
            down += 1.0
    if down == 0:
        return 0, finish
    else:
        logging.info("up: " + str(up))
        # logging.info("down: " + str(down))
        return up/down, finish


if __name__ == '__main__':

    try:
        root_dir = '/home/wby/wby/wby_outputs/train/sgdet_train/train_img_predinfo_txt_noconstraint'
        dqn = DQN()
        # dqn = torch.load('')
        file_list = os.listdir(root_dir)
        times = 0
        finish = False

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
            step = 0
            true_num = 0
            pic_num = 0
            for line in f:
                words = line.split(' ')
                pic_id = int(words[0])
                box_id = int(words[1])
                box_reward = float(words[2])
                box_reward = int(box_reward)
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
                    if true_num >= 1:
                        logging.info("true num: " + str(true_num))
                        pic_num += 1
                        num = len(eig_vec_all)
                        c = np.zeros((1, 64))
                        eig_mat = np.array(eig_vec_all)
                        # print(eig_mat)
                        dqn.set_data(eig_mat, last_pic)
                        last_score = 0
                        for i in range(35 - 1):
                            action = dqn.choose_action(c)
                            # logging.info("aciton: " + str(action))
                            c_last = c.copy()
                            c[0, action] = 1
                            if i != 35 - 2:
                                # print(c)
                                # print(c_true)
                                score, finish = get_score(c, c_true)
                                reward = last_score - score
                                last_score = score
                                # logging.info("reward: " + str(reward))
                                logging.info("score: " + str(score))
                            else:
                                score, finish = get_score(c, c_true)
                                reward = score
                            dqn.store_transition(c_last, action, reward, c, last_pic)
                        if times > 0:
                            dqn.learn()
                            finish = dqn.print_loss()
                            dqn.save_model()
                            if finish:
                                break
                    eig_vec_all.clear()
                    c_true.clear()
                    true_num = 0
                    # print(last_pic)
                    logging.info("pic: " + str(last_pic))
                    logging.info("---------------------------")
                    if pic_num >= 100:
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
            times += 1
            if times >= 100:
                break
        dqn.save_model()

        # 测试代码
        times = 0
        for file in file_list:
            times += 1
            path = root_dir + '/' + file
            f = open(path, mode='r')
            last_pic = -1
            eig_vec_all = []
            c_true = []
            step = 0
            reward_all = []
            pic_num = 0
            true_num = 0
            for line in f:
                words = line.split(' ')
                pic_id = int(words[0])
                box_id = int(words[1])
                box_reward = float(words[2])
                box_reward = int(box_reward)
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
                if last_pic < pic_id:
                    if true_num >= 1:
                        pic_num += 1
                        num = len(eig_vec_all)
                        c = np.zeros((1, num))
                        eig_mat = np.array(eig_vec_all)
                        dqn.set_data(eig_mat, last_pic)
                        last_score = 0
                        Q = np.zeros((1, 35), dtype=float)
                        c_all = np.zeros((34, num))
                        for i in range(35):
                            action = dqn.choose_action_test(c)
                            q = dqn.get_q(c, action)
                            c_last = c
                            c[0, action] = 1
                            c_all[i + 1:i + 2, :] = c
                            Q[0, i] = q
                        s_max = - sys.float_info.max
                        max_i = 0
                        print(Q)
                        for i in range(1, 35):
                            s_i = Q[0, i] + (1 - GAMMA) * (np.sum(Q[:, i + 1:-1], axis=1)) - GAMMA * Q[:, -1]
                            if s_i > s_max:
                                max_i = i
                                s_max = s_i
                                print(s_max)

                        c_max = c_all[max_i:max_i + 1, :]
                        print(c_max)
                        print(c_true)
                        reward = get_score(c_max, c_true)
                        reward_all.append(reward)
                        print(reward)
                        logging.info("reward: " + str(reward))
                    eig_vec_all.clear()
                    c_true.clear()
                    true_num = 0
                    if box_reward == 1:
                        true_num += 1
                    if pic_num >= 10:
                        break
                    eig_vec_all.append(eig_vec)
                    c_true.append(box_reward)
                    last_pic = pic_id

                else:
                    if box_reward == 1:
                        true_num += 1
                    eig_vec_all.append(eig_vec)
                    c_true.append(box_reward)
            f.close()
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

