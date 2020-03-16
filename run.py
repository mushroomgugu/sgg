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
Tensor = FloatTensor

# 参数
BATCH_SIZE = 64
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 34000
Q_NETWORK_ITERATION = 100


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

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

    def forward(self, x, d, s):
        return self.w_1.mm(x.t()) + x.mm(self.w_2).mm(d.t()) - x.mm(self.w_3).mm(F.tanh(s.t())) + self.bias

    def reset_parameters(self):
        stdv_1 = 1. / math.sqrt(self.w_1.size(0))
        self.w_1.data.uniform_(-stdv_1, stdv_1)
        stdv_2 = 1. / math.sqrt(self.w_2.size(0))
        self.w_2.data.uniform_(-stdv_2, stdv_2)
        stdv_3 = 1. / math.sqrt(self.w_3.size(0))
        self.w_3.data.uniform_(-stdv_3, stdv_3)

# 模型
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(4, 40)
        self.layer1.weight.data.normal_(-0.0001, 0.0001)
        self.layer1.bias.data.normal_(-0.01, 0.01)
        self.layer2 = nn.Linear(8272, 1024)
        self.layer3 = nn.Linear(1024, 1, False)
        self.layer4 = MyLayer(4136, 1)

    def forward(self, x, c, a):
        # print(c)
        # v
        num = np.shape(c)[1]
        if torch.cuda.is_available():
            lay1_x = torch.zeros(1, 4136, requires_grad=True).cuda()
            w = torch.zeros(1, num).cuda()
            chosen = torch.zeros(num, 4136, requires_grad=True).cuda()
            chosen_w = torch.zeros(1, num, requires_grad=True).cuda()
            chosen_num = 0
            action_ = torch.zeros(1, 4136, requires_grad=True).cuda()
        else:
            lay1_x = torch.zeros(1, 4136, requires_grad=True)
            w = torch.zeros(1, num, requires_grad=True)
            chosen = torch.zeros(num, 4136, requires_grad=True)
            chosen_w = torch.zeros(1, num, requires_grad=True)
            chosen_num = 0
            action_ = torch.zeros(1, 4136, requires_grad=True)

        # action:
        a = int(a)
        xy = x[a:a + 1, 4096:4100]
        if torch.cuda.is_available():
            xy = torch.tensor(xy, dtype=torch.float32, requires_grad=True).cuda()
        else:
            xy = torch.tensor(xy, dtype=torch.float32, requires_grad=True)
        xy = self.layer1(xy)
        action_box = x[a:a + 1, :4096]
        if torch.cuda.is_available():
            action_box = torch.tensor(torch.from_numpy(action_box), dtype=torch.float32, requires_grad=True).cuda()
            box_vec = torch.cat((action_box, xy), 1).cuda()
        else:
            action_box = torch.tensor(torch.from_numpy(action_box), dtype=torch.float32, requires_grad=True)
            box_vec = torch.cat((action_box, xy), 1)
        action_ = box_vec

        has_chosen = False
        for i in range(num):
            xy = x[i:i + 1, 4096:4100]
            if torch.cuda.is_available():
                xy = torch.tensor(xy, dtype=torch.float32, requires_grad=True).cuda()
            else:
                xy = torch.tensor(xy, dtype=torch.float32, requires_grad=True)
            xy = self.layer1(xy)
            action_box = x[i:i + 1, :4096]
            if torch.cuda.is_available():
                action_box = torch.tensor(torch.from_numpy(action_box), dtype=torch.float32, requires_grad=True).cuda()
                box_vec = torch.cat((action_box, xy), 1).cuda()
            else:
                action_box = torch.tensor(torch.from_numpy(action_box), dtype=torch.float32, requires_grad=True)
                box_vec = torch.cat((action_box, xy), 1)
            if box_vec.shape != action_.shape:
                print(box_vec.shape)
                print(action_.shape)
            box_vec_a = torch.cat((box_vec, action_), 1)
            if i != 0:
                lay1_x = torch.cat((lay1_x, box_vec), 0)
            else:
                lay1_x = box_vec
            # w
            box_vec_ = F.tanh(self.layer2(box_vec_a))
            box_vec_ = self.layer3(box_vec_)
            if i != 0:
                w = torch.cat((w, box_vec_), 1)
            else:
                w = box_vec_
            if c[0, i] == 1:
                if has_chosen:
                    chosen = torch.cat((chosen, box_vec), 0)
                    chosen_w = torch.cat((chosen_w, box_vec_), 1)
                else:
                    chosen = box_vec
                    chosen_w = box_vec_
                    has_chosen = True
                chosen_num += 1
        w_softmax = F.softmax(w, dim=0)
        chosen_w_true = chosen_w[0:1, :chosen_num]
        c_w_softmax = F.softmax(chosen_w_true, dim=0)
        chosen_true = chosen[:chosen_num, :]
        d = w_softmax.mm(lay1_x)
        if chosen_num != 0:
            s = c_w_softmax.mm(chosen_true)
        else:
            if torch.cuda.is_available():
                s = torch.zeros(1, 4136).cuda()
            else:
                s = torch.zeros(1, 4136)
        output = self.layer4(action_, d, s)
        return output

# DQN
class DQN:
    def __init__(self):
        super(DQN, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eval_net, self.target_net = QNetwork(), QNetwork()
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

    def choose_action(self, c):
        if np.random.randn() <= EPISILO:  # greedy policy
            max_q = - sys.float_info.max
            action_n = -1
            for i in range(self.box_num):
                if c[0, i] != 1:
                    action_value = self.eval_net(self.all_box, c, i)
                    if torch.cuda.is_available():
                        action_value_cpu = action_value.cpu()
                        # print("choose_action")
                        # print(action_value_cpu.detach().numpy())
                        if action_value_cpu.detach().numpy()[0, 0] > max_q:
                            action_n = i
                            max_q = action_value_cpu.detach().numpy()[0, 0]
                    else:

                        if action_value.detach().numpy()[0, 0] > max_q:
                            action_n = i
                            max_q = action_value.detach().numpy()[0, 0]
            action = action_n
        else:  # random policy
            while True:
                action = np.random.randint(0, self.box_num)
                if c[0, action] == 0:
                    break
        return action

    def choose_action_test(self, c):
        max_q = - sys.float_info.max
        action_n = -1
        for i in range(self.box_num):
            if c[0, i] != 1:
                action_value = self.eval_net(self.all_box, c, i)
                if torch.cuda.is_available():
                    action_value_cpu = action_value.cpu()
                    # print("choose_action")
                    # print(action_value_cpu.detach().numpy())
                    if action_value_cpu.detach().numpy()[0, 0] > max_q:
                        action_n = i
                        max_q = action_value_cpu.detach().numpy()[0, 0]
                else:

                    if action_value.detach().numpy()[0, 0] > max_q:
                        action_n = i
                        max_q = action_value.detach().numpy()[0, 0]
        # action = action_n
        return action_n

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

        # q_eval
        for k in range(np.shape(batch_memory)[0]):
            # print(batch_pic_id[k, 0])
            # print(int(batch_pic_id[k, 0][0, 0]))

            all_box = self.all_box_list[int(batch_pic_id[k, 0])]
            q_eval = self.eval_net(all_box, batch_state[k:k + 1, :], batch_action[k, 0])
            q_next_max = - sys.float_info.max
            box_num = np.shape(all_box)[0]
            for j in range(box_num):
                if batch_next_state[k:k + 1, j] == 0:
                    # print('next')
                    # print(j)
                    q_next = self.target_net(all_box, batch_next_state[k:k + 1, :], j)
                    # print(q_next.detach().numpy())
                    if torch.cuda.is_available():
                        if q_next_max < q_next.cpu().detach().numpy()[0, 0]:
                            if torch.cuda.is_available():
                                q_next_max = q_next.cpu().detach().numpy()[0, 0]
                            else:
                                q_next_max = q_next.cpu().detach().numpy()[0, 0]
                            q_next_max_true = q_next
                    else:
                        if q_next_max < q_next.detach().numpy()[0, 0]:
                            if torch.cuda.is_available():
                                q_next_max = q_next.cpu().detach().numpy()[0, 0]
                            else:
                                q_next_max = q_next.detach().numpy()[0, 0]
                            q_next_max_true = q_next
            q_target = batch_reward[k, 0] + GAMMA * q_next_max_true
            loss = self.loss_func(q_eval, q_target)
            self.loss_arr.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def get_q(self, state, action):
        q_tensor = self.eval_net(self.all_box, state, action)
        if torch.cuda.is_available():
            return q_tensor.cpu().detach().numpy()[0, 0]
        else:
            return q_tensor.detach().numpy()[0, 0]

    def print_loss(self):
        print(self.loss_arr)
        finish = True
        for loss in self.loss_arr:
            if torch.cuda.is_available():
                if loss.cpu().detach().numpy() > 0.01:
                    finish = False
            else:
                print(loss)
                print(loss.detach())
                print(loss.detach().numpy())
                if loss.detach().numpy() > 0.01:
                    finish = False
        self.loss_arr.clear()
        return finish

    def save_model(self):
        torch.save(self.eval_net, 'eval_net.pkl')
        torch.save(self.target_net, 'target_net.pkl')


def get_score(c, c_true):
    up = 0.0
    down = 0.0
    finish = True
    for i in range(np.shape(c)[1]):
        if c[0, i] == c_true[i] and c[0, i] == 1:
            up += 1.0
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
        file_list = os.listdir(root_dir)
        times = 0
        finish = False

        # 训练
        while True:
            if times > 0:
                logging.info("================================")
            rand_txt = random.choice(file_list)
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
                            if finish:
                                break
                    eig_vec_all.clear()
                    c_true.clear()
                    true_num = 0
                    # print(last_pic)
                    logging.info("pic: " + str(last_pic))
                    logging.info("---------------------------")
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
            if finish:
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

