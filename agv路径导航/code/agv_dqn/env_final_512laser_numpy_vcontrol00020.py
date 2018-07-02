import numpy as np
import random
import sys
import cv2
import time
from RVO_vcontrol import RVO_update, reach, compute_V_des, reach
map_img = cv2.imread('./data/out_40.png', -1)

from libenv.example import Env

###########################################

N = 40
N_A = 4
waittime = 1
debug = 0
import math
TIME = 30
A=1000.0
T=0.3
V_MAX = 2.0
DEBUG_F = 0
show = True
SCAN_DEGREE = 180
EVERY_DEGREE =12
EVERY_STEP = 0.4
Pi =3.1415926
Pi_pre = Pi/180.0
LASER_RADIUS = 4
AGV_RADIUS = 0.5
HALF_AGV_RADIUS = 0.25
DEGREE_NUM = int(SCAN_DEGREE / EVERY_DEGREE) + 1
LASER_NUM = int(LASER_RADIUS / EVERY_STEP)
EVERY_RADIUS = LASER_RADIUS/float(LASER_NUM)
LASER_SIZE = int((SCAN_DEGREE / EVERY_DEGREE + 1) * LASER_RADIUS / EVERY_STEP)
class MyENV:
    def __init__(self, num_agent, winname='render', filename='log'):

        self.observation_space_shape = [6]
        self.laser_size = LASER_SIZE
        self.action_space_n = 5
        self.num_agent = num_agent

        self.t = 0
        self.show = False

        self._env = Env(map_img, num_agent)
        self.start_port = self._env.GetStartPort()
        self.target_port = self._env.GetEndPort()
        self.obstacle_port = self._env.GetObsPort()

        self.destination = None

        self.winname = winname

        self.filename = filename
        # self.file = open(self.filename,mode='w')

        self.theta_dir = [0] * num_agent

        self.use_sp_cmd = [0] * num_agent
        self.rotate_flag = [0] * num_agent
        self.f_use_sp_cmd = [0] * num_agent
        #self.endpoint = []

        self.cmd_state = np.zeros((num_agent,), dtype='int32')
        self.actions = np.zeros((num_agent, 4), dtype='float32')
        self.actions_second = np.zeros((num_agent, 4), dtype='float32')

        self.future_actions = np.zeros((num_agent, 4), dtype='float32')
        self.future_actions_second = np.zeros((num_agent, 4), dtype='float32')
        self.laser_point = np.zeros((num_agent, LASER_SIZE, 2), dtype='float32')
        self.laser_mask = np.zeros((num_agent, LASER_SIZE, 1), dtype='float32')
        self.obstacle_np = np.array(self.obstacle_port).reshape(201,2)
        self.future_states_300 = np.zeros((num_agent, 300, 2), dtype='float32')
        #self.done = np.zeros((self.num_agent, 1), dtype='float32')

        self.mask = np.zeros((N, N), dtype='uint16')
        for x, y in self.target_port:
            self.mask[int(y / 500), int(x / 500)] = 500
        for x, y in self.obstacle_port:
            self.mask[int(y / 500), int(x / 500)] = 500
        self.all_laser_radius = np.zeros((self.num_agent, LASER_NUM), dtype='float32')
        self.all_laser_degree = np.zeros((self.num_agent, DEGREE_NUM), dtype='float32')
        laser_radius = np.arange(LASER_NUM)
        laser_degree = np.arange(DEGREE_NUM) * EVERY_DEGREE-90
        for mm in range(self.num_agent):
            self.all_laser_radius[mm,:] = laser_radius.copy()+1
            self.all_laser_degree[mm, :] = laser_degree.copy()
        self.all_laser_radius = self.all_laser_radius*EVERY_RADIUS
        print('initial')

    def reset(self):

        self.t = 0
        self.show = False

        s_idx = random.sample(range(0, len(self.start_port)), self.num_agent)
        d_idx = random.sample(range(0, len(self.target_port)), self.num_agent)

        _state = self._env.Reset(s_idx, d_idx)

        self.state = np.zeros((self.num_agent, self.observation_space_shape[0]),
                              dtype='float32')  # x0, y0, x1, y1, xclean, yclean
        self.laser_mask = np.zeros((self.num_agent, LASER_SIZE, 1), dtype='float32')
        #self.state= _state.copy()
        self.state[:, 0] = _state[:, 0]*1
        self.state[:, 1] = _state[:, 1]*1
        state_theta_tmp = _state[:, 2].copy()
        state_theta_tmp = state_theta_tmp+360.0
        state_theta_tmp = state_theta_tmp%360.0
        self.state[:, 2] = state_theta_tmp.copy()
        self.state[:, 3] = _state[:, 3]*1

        self.state[:, 4] = [self.target_port[i][0] for i in d_idx]
        self.state[:, 5] = [self.target_port[i][1] for i in d_idx]
        theta_0_360 = _state[:, 2] + 360.0
        theta_0_360 = theta_0_360 % 360.0
        #start_time = time.time()
        X = [[self.state[i, 0]/1000.0,self.state[i, 1]/1000.0] for i in range(self.num_agent)]
        goal = [[self.state[i, 4] / 1000.0, self.state[i, 5] / 1000.0] for i in range(self.num_agent)]
        V = [[self.state[i, 3] / 1000.0*math.cos(self.state[i, 2]*Pi/180.0), self.state[i, 3] / 1000.0*math.sin(self.state[i, 2]*Pi/180.0)] for i in range(self.num_agent)]
        V_Theta = [theta_0_360[i] for i in range(self.num_agent)]
        #V_des = compute_V_des(X, goal, V_MAX)
        # compute the optimal vel to avoid collision
        #v_suit_flag = RVO_update(X, V_des, V, V_Theta, AGV_RADIUS)
        #end_time = time.time()
        #during_time = end_time - start_time
        #for aa in range(self.num_agent):
        #    laser_radius[]
        car_information = 1
        #radius
        multi_ab = np.zeros((self.num_agent, LASER_NUM, DEGREE_NUM,2),dtype='float32')
        init_xy = np.zeros((self.num_agent, LASER_NUM, DEGREE_NUM,2),dtype='float32')
        #degree_theta = np.zeros((self.num_agent, DEGREE_NUM), dtype='float32')
        #for ii in range(self.num_agent):
        #    degree_theta[ii,:] = _state[ii, 2]*1
        #multi_ab_1 = np.zeros((self.num_agent, LASER_NUM, DEGREE_NUM), dtype='float32')
        #start_time = time.time()

        degree_real = theta_0_360.reshape(20, 1) + self.all_laser_degree
        co = np.cos(degree_real * Pi / 180.0)
        si = np.sin(degree_real * Pi / 180.0)
        for ii in range(self.num_agent):
            multi_ab[ii,:,:,0] = self.all_laser_radius[ii,:].reshape(LASER_NUM,1)*co[ii,:].reshape(1,DEGREE_NUM)
            multi_ab[ii, :, :,1] = self.all_laser_radius[ii, :].reshape(LASER_NUM, 1) * si[ii, :].reshape(1, DEGREE_NUM)
            init_xy[ii, :, :, 0] = _state[ii, 0]/1000.0
            init_xy[ii, :, :, 1] = _state[ii, 1] / 1000.0
        multi_ab = multi_ab+init_xy
        obtacles_infor = np.zeros((self.num_agent, self.num_agent+201,LASER_SIZE,2), dtype='float32')
        _state_do = np.concatenate((_state[:,0:2].copy(),self.obstacle_np), axis=0)
        states_car = _state_do.reshape(self.num_agent+201,1,2)
        #if 1 : #for static obtacles
        obstacle_np = abs(self.obstacle_np.copy()/1000.0-_state_do[:,0:2].reshape(self.num_agent+201,1,2)/1000.0)
        obstacle_np_1 = obstacle_np[:,:,0] + obstacle_np[:,:,1]
        for ii in range(self.num_agent):
            obtacles_infor[ii,:,:,:] = multi_ab.reshape(self.num_agent,LASER_SIZE,2)[ii,:,:].reshape(1,LASER_SIZE,2)-_state_do[:,0:2].reshape(self.num_agent+201,1,2)/1000.0
            #obs_dis = obstacle_np_1[ii,:].copy()
            #obs_dis_201 = obstacle_np_1[ii,:]<=4

            #obs_index = np.argwhere(obs_dis_201 == True)
            #multi_ab[ii,:,:,:].reshape(LASER_SIZE,2)-
            tmp_o = obtacles_infor[ii,:,:,:].copy()
            aa = tmp_o.shape[0]*tmp_o.shape[1]
            tmp_o_0 = tmp_o.reshape(aa,2)
            tmp_1 = tmp_o_0[:,0]
            tmp_2 = tmp_o_0[:,1]
            i_mask = (tmp_1<=0.25) & (tmp_1>=-0.25) & (tmp_2<=0.25) & (tmp_2>=-0.25)
            i_index = np.argwhere(i_mask==True)
            if i_index.size>0:
                for nn in i_index:
                    index_mask = nn%LASER_SIZE
                    self.laser_mask[ii,index_mask,0] = 1
        #end_time = time.time()
        #during_time = end_time-start_time
        #multi_ab_co = self.all_laser_radius.reshape(self.num_agent,LASER_NUM,1)*co.reshape(1,self.num_agent,DEGREE_NUM)
        #multi_ab_sin = self.all_laser_radius*si
        #for i in range(self.num_agent):
        #    self.laser_point[i,:,:] ,self.laser_mask[i,:,:] = self.circle_laser(i, self.state[:, 0],self.state[:, 1],self.state[:, 2], LASER_RADIUS)
            #aa,bb = self.circle_laser(i, self.state[:, 0],self.state[:, 1],self.state[:, 2], LASER_RADIUS)
        #end_time = time.time()
        #during_time = end_time-start_time
        if 0:
            print('collid')

            render_img = self._env.Render(0)
            # aaa = cv2.resize(render*1,(500,500))
            lm = render_img.reshape(N * 50, N * 50, 3) * 1

            #if 1:
                #for i_a in range(self.num_agent):
            i_a =0
            cv2.circle(lm, (int(_state[i_a][0] / 10), int(_state[i_a][1] / 10)), 10, (255, 0, 0), -1)
            for ii in range(LASER_NUM):
                for jj in range(DEGREE_NUM):
                    #cv2.circle(lm, (int(multi_ab[i_a][ii][jj][0] * 100), int(multi_ab[i_a][ii][jj][1] * 100)), 10,(0, 255, 0), -1)
                    if self.laser_mask[i_a][DEGREE_NUM*ii+jj][0] ==1:
                        cv2.circle(lm, (int(multi_ab[i_a][ii][jj][0] * 100), int(multi_ab[i_a][ii][jj][1] * 100)), 10,(0, 255, 0), -1)
                    else:
                        cv2.circle(lm, (int(multi_ab[i_a][ii][jj][0] * 100), int(multi_ab[i_a][ii][jj][1] * 100)), 10, (0, 0, 255), -1)
            if 0:

                for ii in range(LASER_SIZE):
                    if self.laser_mask[i_a][ii][0] ==1:
                        cv2.circle(lm, (int(self.laser_point[i_a][ii][0] * 100), int(self.laser_point[i_a][ii][1] * 100)), 10,(0, 255, 0), -1)
                    else:
                        cv2.circle(lm, (int(self.laser_point[i_a][ii][0]*100), int(self.laser_point[i_a][ii][1]*100)), 10, (0, 0, 255), -1)

            # lm[yy:(yy + 20), xx:(xx + 20), 1] = 0
            # lm[yy:(yy + 20), xx:(xx + 20), 2] = 1
            for i_a in range(self.num_agent):
                cv2.putText(lm, str(i_a), (int(_state[i_a, 0] / 10) - 20, int(_state[i_a, 1] / 10) + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 6)
                cv2.putText(lm, str(i_a), (int(self.state[i_a, 4] / 10) - 20, int(self.state[i_a, 5] / 10) + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 6)
            lm = cv2.resize(lm, (500, 500))
            cv2.imwrite('/hik/home/xinchao5/agv/RVO_Py_MAS/agv_obtacles/sim_xin_ok/libenv/data/aa.png', lm)

        self.reward = np.zeros((self.num_agent, 1), dtype='float32')
        self.tcmd = [0] * self.num_agent
        self.done = np.zeros((self.num_agent, 1), dtype='float32')
        self.real_done = np.zeros((self.num_agent, 1), dtype='float32')
        self.crash = np.zeros((self.num_agent,), dtype='float32')
        #self.endpoint = range(self.num_agent)*1
        if 0:
            for agv_i in range(self.num_agent):
                self.take_action(agv_i,0)
                future_states_300 = self.get_future_state(agv_i)
        return self.state*1, range(self.num_agent),self.laser_mask*1,multi_ab#,self.future_states_300*1

    def ireset(self, idx):

        self.show = False
        self.laser_mask = np.zeros((self.num_agent, LASER_SIZE, 1), dtype='float32')
        s_idx = random.randrange(0, len(self.start_port))
        d_idx = random.randrange(0, len(self.target_port))

        _state = self._env.iReset(idx, s_idx, d_idx)
        self.state[idx, 0] = _state[idx, 0]*1
        self.state[idx, 1] = _state[idx, 1]*1
        state_theta_tmp = _state[idx, 2].copy()
        state_theta_tmp = state_theta_tmp + 360.0
        state_theta_tmp = state_theta_tmp % 360.0
        self.state[idx, 2] = state_theta_tmp.copy()
        #self.state[idx, 2] = _state[idx, 2]*1
        self.state[idx, 3] = _state[idx, 3]*1

        self.state[idx, 4] = self.target_port[d_idx][0]
        self.state[idx, 5] = self.target_port[d_idx][1]
        theta_0_360 = _state[:, 2] + 360
        theta_0_360 = theta_0_360 % 360
        X = [[self.state[i, 0] / 1000.0, self.state[i, 1] / 1000.0] for i in range(self.num_agent)]
        goal = [[self.state[i, 4] / 1000.0, self.state[i, 5] / 1000.0] for i in range(self.num_agent)]
        V = [[self.state[i, 3] / 1000.0 * math.cos(self.state[i, 2] * Pi / 180.0),
              self.state[i, 3] / 1000.0 * math.sin(self.state[i, 2] * Pi / 180.0)] for i in range(self.num_agent)]
        V_Theta = [theta_0_360[i] for i in range(self.num_agent)]
        #V_des = compute_V_des(X, goal, V_MAX)
        # compute the optimal vel to avoid collision
        #v_suit_flag = RVO_update(X, V_des, V, V_Theta, AGV_RADIUS)
        self.tcmd[idx] = 0
        self.done[idx, 0] = 0
        self.real_done[idx, 0] = 0
        endpoint = []
        endpoint.append(idx)
        self.crash[idx] = 0
        #self.laser_point[idx, :, :], self.laser_mask[idx, :, :] = self.circle_laser(idx, self.state[:, 0], self.state[:, 1],self.state[:, 2], LASER_RADIUS)

        # radius
        multi_ab = np.zeros((self.num_agent, LASER_NUM, DEGREE_NUM, 2), dtype='float32')
        init_xy = np.zeros((self.num_agent, LASER_NUM, DEGREE_NUM, 2), dtype='float32')
        # degree_theta = np.zeros((self.num_agent, DEGREE_NUM), dtype='float32')
        # for ii in range(self.num_agent):
        #    degree_theta[ii,:] = _state[ii, 2]*1
        # multi_ab_1 = np.zeros((self.num_agent, LASER_NUM, DEGREE_NUM), dtype='float32')
        # start_time = time.time()

        degree_real = theta_0_360.reshape(20, 1) + self.all_laser_degree
        co = np.cos(degree_real * Pi / 180.0)
        si = np.sin(degree_real * Pi / 180.0)
        for ii in range(self.num_agent):
            multi_ab[ii, :, :, 0] = self.all_laser_radius[ii, :].reshape(LASER_NUM, 1) * co[ii, :].reshape(1,DEGREE_NUM)
            multi_ab[ii, :, :, 1] = self.all_laser_radius[ii, :].reshape(LASER_NUM, 1) * si[ii, :].reshape(1,DEGREE_NUM)
            init_xy[ii, :, :, 0] = _state[ii, 0] / 1000.0
            init_xy[ii, :, :, 1] = _state[ii, 1] / 1000.0
        multi_ab = multi_ab + init_xy
        obtacles_infor = np.zeros((self.num_agent, self.num_agent + 201, LASER_SIZE, 2), dtype='float32')
        _state_do = np.concatenate((_state[:, 0:2].copy(), self.obstacle_np), axis=0)
        states_car = _state_do.reshape(self.num_agent + 201, 1, 2)
        # if 1 : #for static obtacles
        obstacle_np = abs(
            self.obstacle_np.copy() / 1000.0 - _state_do[:, 0:2].reshape(self.num_agent + 201, 1, 2) / 1000.0)
        obstacle_np_1 = obstacle_np[:, :, 0] + obstacle_np[:, :, 1]
        for ii in range(self.num_agent):
            obtacles_infor[ii, :, :, :] = multi_ab.reshape(self.num_agent, LASER_SIZE, 2)[ii, :, :].reshape(1, LASER_SIZE,2) - _state_do[:,0:2].reshape(self.num_agent + 201, 1, 2) / 1000.0
            # obs_dis = obstacle_np_1[ii,:].copy()
            # obs_dis_201 = obstacle_np_1[ii,:]<=4

            # obs_index = np.argwhere(obs_dis_201 == True)
            # multi_ab[ii,:,:,:].reshape(LASER_SIZE,2)-
            tmp_o = obtacles_infor[ii, :, :, :].copy()
            aa = tmp_o.shape[0] * tmp_o.shape[1]
            tmp_o_0 = tmp_o.reshape(aa, 2)
            tmp_1 = tmp_o_0[:, 0]
            tmp_2 = tmp_o_0[:, 1]
            i_mask = (tmp_1 <= 0.25) & (tmp_1 >= -0.25) & (tmp_2 <= 0.25) & (tmp_2 >= -0.25)
            i_index = np.argwhere(i_mask == True)
            if i_index.size > 0:
                for nn in i_index:
                    index_mask = nn % LASER_SIZE
                    self.laser_mask[ii, index_mask, 0] = 1
        #self.endpoint = endpoint*1
        if 0:
            self.take_action(idx, 0)
            future_states_300 = self.get_future_state(idx)
        # self.state[idx, 48] = _state[idx, 0]
        # self.state[idx, 49] = _state[idx, 1]

        return self.state*1,endpoint,self.laser_mask*1,multi_ab

    def set_action(self, agent_idx, act):
        if self.done[agent_idx] > 0:
            return
        self.crash[agent_idx] = 0
        cur_x = self.state[agent_idx][0]
        cur_y = self.state[agent_idx][1]
        cur_theta = self.state[agent_idx][2]
        cur_v = self.state[agent_idx][3]
        #rotate_flag = self.state[0][-1]
        #render = mm.Render(Debug_Info)
        theta_num = self._pre_compute_theta(cur_theta)

        if act == 1:
            self.use_sp_cmd[agent_idx] = 1
            a = A
            p = 2000

            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                self.actions[agent_idx] = [cur_x + p, 0, 0, cur_theta]

            if theta_num == -3 or theta_num == 1:
                # p = 8000
                self.actions[agent_idx] = [cur_y + p, 0, 0, cur_theta]

            if theta_num == -2 or theta_num == 2:
                self.actions[agent_idx] = [cur_x - p, 0, 0, cur_theta]

            if theta_num == -1 or theta_num == 3:
                # p = 8000
                self.actions[agent_idx] = [cur_y - p, 0, 0, cur_theta]

            self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))

                # a1 = a.plot(time_t*mmmm+i, states[0][3], 'ro--', label='line 1')
                # b1 = b.plot(time_t*mmmm+i, aa, 'ro--', label='line 1')
                # render = mm.Render(0)


        if act == 0:
            self.use_sp_cmd[agent_idx] = 0
            a = A
            # next_v = cur_v + a * T
            p = cur_v * cur_v / (2 * a)
            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                p_p = cur_x + p
                aa = math.ceil((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, cur_theta]

            if theta_num == -3 or theta_num == 1:
                # p = 8000
                p_p = cur_y + p
                aa = math.ceil((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, cur_theta]

            if theta_num == -2 or theta_num == 2:
                p_p = cur_x - p
                aa = math.floor((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, cur_theta]

            if theta_num == -1 or theta_num == 3:
                # p = 8000
                p_p = cur_y - p
                aa = math.floor((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, cur_theta]
            # action = [p, next_v, 200, cur_theta]
            self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))

        # action = [cur_x + p, 0, 0, cur_theta]

        if act == 2:
            #self.use_sp_cmd[agent_idx] = 2
            self.use_sp_cmd[agent_idx] = 2
            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x + p
                aa = math.ceil((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                # p = 8000
                self.actions[agent_idx] = [p_p, 0, 0, cur_theta]

            if theta_num == -3 or theta_num == 1:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                p_p =cur_y + p
                aa = math.ceil((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, cur_theta]

            if theta_num == -2 or theta_num == 2:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                p_p = cur_x - p
                aa = math.floor((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, cur_theta]

            if theta_num == -1 or theta_num == 3:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                p_p = cur_y - p
                aa = math.floor((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, cur_theta]

            self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))

                # a1 = a.plot(time_t*mmmm+i, states[0][3], 'ro--', label='line 1')
                # b1 = b.plot(time_t*mmmm+i, aa, 'ro--', label='line 1')
                # render = mm.Render(0)

        #if act == 3:
        if act == 3:
            self.use_sp_cmd[agent_idx] = 3
            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x + p
                aa = math.ceil((p_p - 250) / 500.0)
                p_p = aa*500.0+250
                self.actions[agent_idx] = [p_p, 0, 0, theta_num * 90]  # special

                if DEBUG_F:
                    actions_once = np.zeros((1, 5), dtype='float32')
                    actions_once[0,0:4] = self.actions[agent_idx].reshape(1, -1) * 1
                    actions_once[0, 4] = -1
                    yyy = self._env.Dryrun2(agent_idx, actions_once)
                    mid_num = np.where(yyy[:, 2] == 1)[0][0]
                    self.actions[agent_idx][0] = yyy[mid_num, 0] * 1

            if theta_num == -3 or theta_num == 1:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                p_p = cur_y + p
                aa = math.ceil((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, theta_num * 90]  # special
                if DEBUG_F:
                    actions_once = np.zeros((1, 5), dtype='float32')
                    actions_once[0, 0:4] = self.actions[agent_idx].reshape(1, -1) * 1
                    actions_once[0, 4] = -1
                    yyy = self._env.Dryrun2(agent_idx, actions_once)
                    mid_num = np.where(yyy[:, 2] == 1)[0][0]
                    self.actions[agent_idx][0] = yyy[mid_num, 1] * 1

            if theta_num == -2 or theta_num == 2:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x - p
                aa = math.floor((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, theta_num * 90]  # special
                if DEBUG_F:
                    actions_once = np.zeros((1, 5), dtype='float32')
                    actions_once[0, 0:4] = self.actions[agent_idx].reshape(1, -1) * 1
                    actions_once[0, 4] = -1
                    yyy = self._env.Dryrun2(agent_idx, actions_once)
                    mid_num = np.where(yyy[:, 2] == 1)[0][0]
                    self.actions[agent_idx][0] = yyy[mid_num, 0] * 1

            if theta_num == -1 or theta_num == 3:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_y - p
                aa = math.floor((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, theta_num * 90]  # special
                if DEBUG_F:
                    actions_once = np.zeros((1, 5), dtype='float32')
                    actions_once[0, 0:4] = self.actions[agent_idx].reshape(1, -1) * 1
                    actions_once[0, 4] = -1
                    yyy = self._env.Dryrun2(agent_idx, actions_once)
                    mid_num = np.where(yyy[:, 2] == 1)[0][0]
                    self.actions[agent_idx][0] = yyy[mid_num, 1] * 1

            self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))
        if act == 4:
            self.use_sp_cmd[agent_idx] = 4
            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x + p
                aa = math.ceil((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, theta_num * 90]  # special
                if DEBUG_F:
                    actions_once = np.zeros((1, 5), dtype='float32')
                    actions_once[0, 0:4] = self.actions[agent_idx].reshape(1, -1) * 1
                    actions_once[0, 4] = -1
                    yyy = self._env.Dryrun2(agent_idx, actions_once)
                    mid_num = np.where(yyy[:, 2] == 1)[0][0]
                    self.actions[agent_idx][0] = yyy[mid_num, 0] * 1

            if theta_num == -3 or theta_num == 1:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                p_p = cur_y + p
                aa = math.ceil((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, theta_num * 90]  # special
                if DEBUG_F:
                    actions_once = np.zeros((1, 5), dtype='float32')
                    actions_once[0, 0:4] = self.actions[agent_idx].reshape(1, -1) * 1
                    actions_once[0, 4] = -1
                    yyy = self._env.Dryrun2(agent_idx, actions_once)
                    mid_num = np.where(yyy[:, 2] == 1)[0][0]
                    self.actions[agent_idx][0] = yyy[mid_num, 1] * 1

            if theta_num == -2 or theta_num == 2:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x - p
                aa = math.floor((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, theta_num * 90]  # special
                if DEBUG_F:
                    actions_once = np.zeros((1, 5), dtype='float32')
                    actions_once[0, 0:4] = self.actions[agent_idx].reshape(1, -1) * 1
                    actions_once[0, 4] = -1
                    yyy = self._env.Dryrun2(agent_idx, actions_once)
                    mid_num = np.where(yyy[:, 2] == 1)[0][0]
                    self.actions[agent_idx][0] = yyy[mid_num, 0] * 1

            if theta_num == -1 or theta_num == 3:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_y - p
                aa = math.floor((p_p - 250) / 500.0)
                p_p = aa * 500.0 + 250
                self.actions[agent_idx] = [p_p, 0, 0, theta_num * 90]  # special
                if DEBUG_F:
                    actions_once = np.zeros((1, 5), dtype='float32')
                    actions_once[0, 0:4] = self.actions[agent_idx].reshape(1, -1) * 1
                    actions_once[0, 4] = -1
                    yyy = self._env.Dryrun2(agent_idx, actions_once)
                    mid_num = np.where(yyy[:, 2] == 1)[0][0]
                    self.actions[agent_idx][0] = yyy[mid_num, 1] * 1
            #self.actions[agent_idx] = [p_p, 0, 0, theta_num*90]  #special


            self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))

    def take_future_action(self, agent_idx, act):
        #if self.done[agent_idx] > 0:
        #    return

        cur_x = self.state[agent_idx][0]
        cur_y = self.state[agent_idx][1]
        cur_theta = self.state[agent_idx][2]
        cur_v = self.state[agent_idx][3]
        #rotate_flag = self.state[0][-1]
        #render = mm.Render(Debug_Info)
        theta_num = self._pre_compute_theta(cur_theta)

        if act == 1:
            self.f_use_sp_cmd[agent_idx] = 1
            a = A
            p = 2000

            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                self.future_actions[agent_idx] = [cur_x + p, 0, 0, cur_theta]

            if theta_num == -3 or theta_num == 1:
                # p = 8000
                self.future_actions[agent_idx] = [cur_y + p, 0, 0, cur_theta]

            if theta_num == -2 or theta_num == 2:
                self.future_actions[agent_idx] = [cur_x - p, 0, 0, cur_theta]

            if theta_num == -1 or theta_num == 3:
                # p = 8000
                self.future_actions[agent_idx] = [cur_y - p, 0, 0, cur_theta]
            #self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))

                # a1 = a.plot(time_t*mmmm+i, states[0][3], 'ro--', label='line 1')
                # b1 = b.plot(time_t*mmmm+i, aa, 'ro--', label='line 1')
                # render = mm.Render(0)


        if act == 0:
            self.f_use_sp_cmd[agent_idx] = 0
            a = A
            # next_v = cur_v + a * T
            p = cur_v * cur_v / (2 * a)
            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                self.future_actions[agent_idx] = [cur_x + p, 0, 0, cur_theta]

            if theta_num == -3 or theta_num == 1:
                # p = 8000
                self.future_actions[agent_idx] = [cur_y + p, 0, 0, cur_theta]

            if theta_num == -2 or theta_num == 2:
                self.future_actions[agent_idx] = [cur_x - p, 0, 0, cur_theta]

            if theta_num == -1 or theta_num == 3:
                # p = 8000
                self.future_actions[agent_idx] = [cur_y - p, 0, 0, cur_theta]
            # action = [p, next_v, 200, cur_theta]
            #self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))

        # action = [cur_x + p, 0, 0, cur_theta]

        if act == 2:
            #self.use_sp_cmd[agent_idx] = 2
            self.f_use_sp_cmd[agent_idx] = 2
            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                self.future_actions[agent_idx] = [cur_x + p, 0, 0, cur_theta]

            if theta_num == -3 or theta_num == 1:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                self.future_actions[agent_idx] = [cur_y + p, 0, 0, cur_theta]

            if theta_num == -2 or theta_num == 2:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                self.future_actions[agent_idx] = [cur_x - p, 0, 0, cur_theta]

            if theta_num == -1 or theta_num == 3:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                self.future_actions[agent_idx] = [cur_y - p, 0, 0, cur_theta]

            #self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))

                # a1 = a.plot(time_t*mmmm+i, states[0][3], 'ro--', label='line 1')
                # b1 = b.plot(time_t*mmmm+i, aa, 'ro--', label='line 1')
                # render = mm.Render(0)

        #if act == 3:
        if act == 3:
            self.f_use_sp_cmd[agent_idx] = 3
            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x + p

            if theta_num == -3 or theta_num == 1:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                p_p = cur_y + p

            if theta_num == -2 or theta_num == 2:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x - p

            if theta_num == -1 or theta_num == 3:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_y - p
            self.future_actions[agent_idx] = [p_p, 0, 0, theta_num*90]  #special
            #self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))
        if act == 4:
            self.use_sp_cmd[agent_idx] = 4
            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x + p

            if theta_num == -3 or theta_num == 1:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                p_p = cur_y + p

            if theta_num == -2 or theta_num == 2:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x - p

            if theta_num == -1 or theta_num == 3:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_y - p
            self.future_actions[agent_idx] = [p_p, 0, 0, theta_num*90]


    def set_action_bp(self,agent_idx):
        #p_ok = self._collid_save(agent_idx, self.state*1)
        self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))
        #return p_ok*1

    def judge_action(self,agent_idx,cs_num,ed_pt):

        p_ok = self._collid_save(agent_idx, self.state, self.future_states_300*1,cs_num,ed_pt)
        #self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))
        return p_ok*1
    def get_future_state(self, agent_idx):
        actions_twice = np.zeros((2, 5), dtype='float32')

        if self.use_sp_cmd[agent_idx] == 0 or self.use_sp_cmd[agent_idx] == 1:
            actions_twice[0,0:4] = self.actions[agent_idx]*1
            actions_twice[0,4] = 1000

            yyy = self._env.Dryrun2(agent_idx, actions_twice[0,:].reshape(1,-1))
            try:
                mid_num = np.where(yyy[:,2]==1)[0][0]
            except:
                print('deadly')
            #self.future_states_300[agent_idx,0:30,0:2] = yyy[0:30,0:2]*1
            self.future_states_300[agent_idx, 0:(mid_num + 1), 0:2] = yyy[0:(mid_num + 1), 0:2] * 1
            self.future_states_300[agent_idx, (mid_num + 1):300, 0:2] = yyy[mid_num, 0:2]*1
        if self.use_sp_cmd[agent_idx] == 2:
            actions_twice[0,0:4] = self.actions[agent_idx] * 1
            actions_twice[0,4] = -1
            yyy = self._env.Dryrun2(agent_idx, actions_twice[0,:].reshape(1,-1))
            mid_num = np.where(yyy[:, 2] == 1)[0][0]
            #if mid_num>0:
            #    print('mm')
            self.future_states_300[agent_idx, 0:(mid_num + 1), 0:2] = yyy[0:(mid_num + 1), 0:2] * 1
            self.future_states_300[agent_idx, (mid_num + 1):300, 0:2] = yyy[mid_num, 0:2]*1

        if self.use_sp_cmd[agent_idx] == 3:
            self.actions_second[agent_idx] = self.actions[agent_idx] * 1
            actions_twice[0,0:4] = self.actions[agent_idx] * 1
            actions_twice[0,4] = -1
            #yyy = self._env.Dryrun2(agent_idx, actions_twice, 1000)

            if (self.actions[agent_idx][3] + 90) == 450:
                self.actions_second[agent_idx][3] = 90
            else:
                self.actions_second[agent_idx][3] = self.actions[agent_idx][3] + 90

            actions_twice[1,0:4] = self.actions_second[agent_idx] * 1
            actions_twice[1,4] = -1
            yyy = self._env.Dryrun2(agent_idx, actions_twice.reshape(2,-1))
            try:
                mid_num = np.where(yyy[:, 2] == 1)[0][0]
                mid_num1 = np.where(yyy[:, 2] == 1)[0][1]
            except:
                print('error')
            if mid_num1>=300 or mid_num>=300:
                print('error ii')
            self.future_states_300[agent_idx, 0:(mid_num1 + 1), 0:2] = yyy[0:(mid_num1 + 1), 0:2] * 1
            self.future_states_300[agent_idx, (mid_num1 + 1):300, 0:2] = yyy[mid_num1, 0:2] * 1
            #self.future_states_300[agent_idx, (mid_num + 1):(mid_num1), 0:2] = yyy[mid_num, 0:2] * 1

            #self.future_states_300[agent_idx, (mid_num1):(300), 0:2] = yyy[mid_num, 0:2] * 1

        if self.use_sp_cmd[agent_idx] == 4:
            self.actions_second[agent_idx] = self.actions[agent_idx] * 1
            actions_twice[0, 0:4] = self.actions[agent_idx] * 1
            actions_twice[0, 4] = -1
            # yyy = self._env.Dryrun2(agent_idx, actions_twice, 1000)

            if (self.actions[agent_idx][3] - 90) == -450:
                self.actions_second[agent_idx][3] = -90
            else:
                # self.actions[idx][3] = self.actions[idx][3]+90
                self.actions_second[agent_idx][3] = self.actions[agent_idx][3] - 90

            actions_twice[1, 0:4] = self.actions_second[agent_idx] * 1
            actions_twice[1, 4] = -1
            yyy = self._env.Dryrun2(agent_idx, actions_twice.reshape(2, -1))
            mid_num = np.where(yyy[:, 2] == 1)[0][0]
            mid_num1 = np.where(yyy[:, 2] == 1)[0][1]
            if mid_num1>=300 or mid_num>=300:
                print('error ii')

            self.future_states_300[agent_idx, 0:(mid_num1 + 1), 0:2] = yyy[0:(mid_num1 + 1), 0:2] * 1
            self.future_states_300[agent_idx, (mid_num1 + 1):300, 0:2] = yyy[mid_num1, 0:2] * 1
        return self.future_states_300*1

    def get_future_state_bp(self, agent_idx):
        actions_twice = np.zeros((2, 5), dtype='float32')
        self.actions_second =  self.actions*1
        #if self.use_sp_cmd[agent_idx] ==0 or self.use_sp_cmd[agent_idx] == 1 or self.use_sp_cmd[agent_idx] ==2:
        yyy  = self._env.Dryrun2(agent_idx, self.actions[agent_idx].reshape(1, -1), 1000)
        future_states = self._env.Dryrun(agent_idx, self.actions[agent_idx].reshape(1, -1), 1000)
        #if self.use_sp_cmd[agent_idx] == 0 or self.use_sp_cmd[agent_idx] == 1 or self.use_sp_cmd[agent_idx] == 2:
        try:
            mid_num = np.where(future_states[:, 2] == 1)[0][0]
        except:
            print('deadly')
        if self.use_sp_cmd[agent_idx] == 0 or self.use_sp_cmd[agent_idx] == 1 or self.use_sp_cmd[agent_idx] == 2:
            self.future_states_300[agent_idx, 0:(mid_num + 1), :] = future_states[0:(mid_num + 1), 0:2] * 1
            for iii in range((mid_num + 1),300):
                self.future_states_300[agent_idx, iii, :] = future_states[mid_num + 1, 0:2] * 1
        else:
            self.future_states_300[agent_idx,0:(mid_num+1),:] = future_states[0:(mid_num+1),0:2]*1

        if self.use_sp_cmd[agent_idx] == 3:

            if (self.actions[agent_idx][3] + 90) == 450:
                self.actions_second[agent_idx][3] = 90
            else:
                self.actions_second[agent_idx][3] = self.actions[agent_idx][3] + 90
            future_states_1 = self._env.Dryrun(agent_idx, self.actions_second[agent_idx].reshape(1, -1), 1000)
            try:
                mid_num_1 = np.where(future_states_1[:, 2] == 1)[0][0]
            except:
                print('deadly')
            if (mid_num_1 + mid_num + 2)<=300:
                self.future_states_300[agent_idx, (mid_num + 1):(mid_num + 1 + mid_num_1 + 1), :] = future_states_1[0:(mid_num_1 + 1), 0:2] * 1
                for jjj in range(mid_num_1+1,300):
                    self.future_states_300[agent_idx, jjj, :] = future_states_1[mid_num_1 ,0:2] * 1
            else:
                self.future_states_300[agent_idx, (mid_num_1 + 1):300, :] = future_states_1[ 0:(299-mid_num_1),0:2] * 1
            #self.future_states_300[agent_idx, (mid_num+1):300, :] = future_states_1[0:(299-mid_num), 0:2] * 1

        if self.use_sp_cmd[agent_idx] == 4:

            if (self.actions[agent_idx][3] - 90) == -450:
                self.actions_second[agent_idx][3] = -90
            else:
                # self.actions[idx][3] = self.actions[idx][3]+90
                self.actions_second[agent_idx][3] = self.actions[agent_idx][3] - 90
            future_states_1 = self._env.Dryrun(agent_idx, self.actions_second[agent_idx].reshape(1, -1), 1000)
            try:
                mid_num_1 = np.where(future_states_1[:, 2] == 1)[0][0]
            except:
                print('deadly')
            if (mid_num_1 + mid_num + 2)<=300:
                self.future_states_300[agent_idx, (mid_num + 1):(mid_num + 1 + mid_num_1 + 1), :] = future_states_1[0:(mid_num_1 + 1), 0:2] * 1
                for jjj in range(mid_num_1+1,300):
                    self.future_states_300[agent_idx, jjj, :] = future_states_1[mid_num_1 ,0:2] * 1
            else:
                self.future_states_300[agent_idx, (mid_num_1 + 1):300, :] = future_states_1[ 0:(299-mid_num_1),0:2] * 1
            #self.future_states_300[agent_idx, (mid_num + 1):300, :] = future_states_1[0:(299 - mid_num), 0:2] * 1
        #self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))
        return self.future_states_300*1

    #self.get_single_future_state(i, action_i, self.future_actions[i])
    def get_single_future_state(self, agent_idx,action_index, action_p):
        actions_twice = np.zeros((2, 5), dtype='float32')
        s_future_states_300 = np.zeros((300, 2), dtype='float32')

        if action_index == 0 or action_index == 1:
            actions_twice[0,0:4] = action_p*1
            actions_twice[0,4] = 1000

            yyy = self._env.Dryrun2(agent_idx, actions_twice[0,:].reshape(1,-1))
            #mid_num = np.where(yyy[:,2]==1)[0][0]
            #s_future_states_300[0:30,0:2] = yyy[0:30,0:2]*1
            try:
                mid_num = np.where(yyy[:, 2] == 1)[0][0]
            except:
                print('deadly')
                # self.future_states_300[agent_idx,0:30,0:2] = yyy[0:30,0:2]*1
            s_future_states_300[ 0:(mid_num + 1), 0:2] = yyy[0:(mid_num + 1), 0:2] * 1
            s_future_states_300[(mid_num + 1):300, 0:2] = yyy[mid_num, 0:2] * 1
            #$self.future_states_300[agent_idx, (mid_num + 1):300, 0:2] = yyy[mid_num, 0:2]*1
        if action_index == 2:
            actions_twice[0,0:4] = action_p * 1
            actions_twice[0,4] = -1
            yyy = self._env.Dryrun2(agent_idx, actions_twice[0,:].reshape(1,-1))
            mid_num = np.where(yyy[:, 2] == 1)[0][0]
            s_future_states_300[0:(mid_num + 1), 0:2] = yyy[0:(mid_num + 1), 0:2] * 1
            s_future_states_300[(mid_num + 1):300, 0:2] = yyy[mid_num, 0:2]*1

        if action_index == 3:
            second_action = action_p * 1
            actions_twice[0,0:4] = action_p * 1
            actions_twice[0,4] = -1
            #yyy = self._env.Dryrun2(agent_idx, actions_twice, 1000)

            if (action_p[3] + 90) == 450:
                second_action[3] = 90
            else:
                second_action[3] = action_p[3] + 90

            actions_twice[1,0:4] = second_action * 1
            actions_twice[1,4] = -1
            yyy = self._env.Dryrun2(agent_idx, actions_twice.reshape(2,-1))
            try:
                mid_num = np.where(yyy[:, 2] == 1)[0][0]
                mid_num1 = np.where(yyy[:, 2] == 1)[0][1]
            except:
                print('error')
            if mid_num1>=300 or mid_num>=300:
                print('error ii')
            s_future_states_300[0:(mid_num1 + 1), 0:2] = yyy[0:(mid_num1 + 1), 0:2] * 1
            s_future_states_300[(mid_num1 + 1):300, 0:2] = yyy[mid_num1, 0:2] * 1
            #self.future_states_300[agent_idx, (mid_num + 1):(mid_num1), 0:2] = yyy[mid_num, 0:2] * 1

            #self.future_states_300[agent_idx, (mid_num1):(300), 0:2] = yyy[mid_num, 0:2] * 1

        if self.use_sp_cmd[agent_idx] == 4:
            second_action = action_p * 1
            actions_twice[0, 0:4] = action_p * 1
            actions_twice[0, 4] = -1
            # yyy = self._env.Dryrun2(agent_idx, actions_twice, 1000)

            if (action_p[3] - 90) == -450:
                second_action[3] = -90
            else:
                # self.actions[idx][3] = self.actions[idx][3]+90
                second_action[3] = action_p[3] - 90

            actions_twice[1, 0:4] = second_action * 1
            actions_twice[1, 4] = -1
            yyy = self._env.Dryrun2(agent_idx, actions_twice.reshape(2, -1))
            mid_num = np.where(yyy[:, 2] == 1)[0][0]
            mid_num1 = np.where(yyy[:, 2] == 1)[0][1]
            if mid_num1>=300 or mid_num>=300:
                print('error ii')

            s_future_states_300[0:(mid_num1 + 1), 0:2] = yyy[0:(mid_num1 + 1), 0:2] * 1
            s_future_states_300[(mid_num1 + 1):300, 0:2] = yyy[mid_num1, 0:2] * 1
        return s_future_states_300*1
    def get_single_future_state_bp(self, agent_idx, action_index, action_p):
        s_future_states_300 = np.zeros((300, 2), dtype='float32')
        actions_second =  action_p*1
        #if self.use_sp_cmd[agent_idx] ==0 or self.use_sp_cmd[agent_idx] == 1 or self.use_sp_cmd[agent_idx] ==2:
        s_future_states = self._env.Dryrun(agent_idx, action_p.reshape(1, -1), 1000)
        #if self.use_sp_cmd[agent_idx] == 0 or self.use_sp_cmd[agent_idx] == 1 or self.use_sp_cmd[agent_idx] == 2:
        s_future_states_300[:,:] = s_future_states[0:300,0:2]*1

        if action_index == 3:
            try:
                mid_num = np.where(s_future_states[:,2]==1)[0][0]
            except:
                print('deadly')
            if (action_p[3] + 90) == 450:
                actions_second[3] = 90
            else:
                actions_second[3] = action_p[3] + 90
            future_states_1 = self._env.Dryrun(agent_idx, actions_second.reshape(1, -1), 1000)
            s_future_states_300[(mid_num+1):300, :] = future_states_1[0:(299-mid_num), 0:2] * 1

        if action_index == 4:
            try:
                mid_num = np.where(s_future_states[:,2]==1)[0][0]
            except:
                print('deadly')
            if (action_p[3] - 90) == -450:
                actions_second[3] = -90
            else:
                # self.actions[idx][3] = self.actions[idx][3]+90
                actions_second[3] = action_p[3] - 90
            future_states_1 = self._env.Dryrun(agent_idx, actions_second.reshape(1, -1), 1000)
            s_future_states_300[(mid_num + 1):300, :] = future_states_1[0:(299 - mid_num), 0:2] * 1
        #self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))
        return s_future_states_300*1



    def step(self):

        # dx0 = (self.state[:,2]//500) - (self.state[:,0]//500)
        # dy0 = (self.state[:,3]//500) - (self.state[:,1]//500)
        # dis0 = abs(dx0) + abs(dy0)
        #dx0 = (self.state[:, 4] // 500) - (self.state[:, 0] // 500)
        #dy0 = (self.state[:,5]//500) - (self.state[:,1]//500)
        #last_rv = last_rv0_v*1
        self.crash = np.zeros((self.num_agent,), dtype='float32')
        self.tcmd = [c+1 for c in self.tcmd] *1
        dis0=[]
        #np.cos(self.state[0]*)
        a1 = np.cos(self.state[:,2]/180.0*Pi)
        a2 = np.sin(self.state[:, 2] / 180.0 * Pi)
        b1 = (self.state[:, 4] - self.state[:, 0])/1000.0
        b2 =  (self.state[:, 5] - self.state[:, 1])/1000
        c1 = (a1*b1+a2*b2)/np.sqrt(b1*b1+b2*b2)
        d0 = np.arccos(c1)
        self.laser_mask = np.zeros((self.num_agent, LASER_SIZE, 1), dtype='float32')
        if 1:
            for i_agv in range(self.num_agent):
            #mm = self._calEuclideanDistance([self.state[0, 4] /1000.0,self.state[0, 5] /1000.0],[self.state[0, 0] /1000.0,self.state[0, 1] /1000.0])
                mm = self._calEuclideanDistance([self.state[i_agv, 4] / 1000.0, self.state[i_agv, 5] / 1000.0],[self.state[i_agv, 0] / 1000.0, self.state[i_agv, 1] / 1000.0])
                dis0.append(mm)
        ttmp = 0
        endpoint = []
        min_d_agv = [1000]*self.num_agent
        last_state = self.state*1

        for _ in range(30):
            _state = self._env.Step()
            for uuuu in range(self.num_agent):
                if _state[uuuu, -2]==2 or _state[uuuu, -2]==3 or _state[uuuu, -2]==1:
                    self.crash[uuuu] = self.crash[uuuu] + 1
            if 0 :
                for xx in range(self.num_agent):
                    yy=[zz for zz in range(self.num_agent)]
                    yy.remove(xx)
                    for uu in yy:
                        tmp_d = self._calEuclideanDistance([_state[xx, 0] / 1000.0, _state[xx, 1] / 1000.0],[_state[uu, 0] / 1000.0, _state[uu, 1] / 1000.0])
                        if tmp_d<min_d_agv[xx]:
                            min_d_agv[xx] = tmp_d
            if 0:
                for  xx in range(self.num_agent):
                    tmp_20_2 =_state[:,0:2].copy()/1000.0 - _state[xx,0:2].reshape(1,2)/1000.0
                    dis_state = abs(tmp_20_2)
                    distance_state = dis_state[:, 0] + dis_state[:, 1]
                    sort_index = list(np.sort(distance_state))
                    if sort_index[1] < min_d_agv[xx]:
                        min_d_agv[xx] = sort_index[1]
            if self.show:
                render_img = self._env.Render(0)
                render_img_0 = render_img.copy()
                # aaa = cv2.resize(render*1,(500,500))
                lm = render_img_0.reshape(N * 50, N * 50, 3)
                for i_a in range(self.num_agent):
                    cv2.putText(lm, str(i_a), (int(_state[i_a, 0] / 10) - 20, int(_state[i_a, 1] / 10) + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 6)
                    cv2.putText(lm, str(i_a), (int(self.state[i_a, 4] / 10) - 20, int(self.state[i_a, 5] / 10) + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 6)
                lm = cv2.resize(lm, (500, 500))
                cv2.imshow(self.winname, lm)
                cv2.waitKey(waittime)
            for idx in range(self.num_agent):
                if (self.use_sp_cmd[idx] == 3 or self.use_sp_cmd[idx] == 4) and idx not in endpoint:
                    if _state[idx, -1] > 0:
                        #if idx ==12:
                        #    print('print debug')
                        if self.cmd_state[idx] > 0:
                            self.cmd_state[idx] = 0
                            self.rotate_flag[idx] = 0
                            endpoint.append(idx)
                        else:
                            self.cmd_state[idx] = 1
                            self.rotate_flag[idx] = 1
                            if self.use_sp_cmd[idx] == 3 :
                                if (self.actions[idx][3] + 90)==450:
                                    self.actions[idx][3] = 90
                                else:
                                    self.actions[idx][3] = self.actions[idx][3]+90
                                self._env.SetAction(idx, self.actions[idx].reshape(1,-1))
                            if self.use_sp_cmd[idx] == 4 :
                                if (self.actions[idx][3] - 90)==-450:
                                    self.actions[idx][3] = -90
                                else:
                                    #self.actions[idx][3] = self.actions[idx][3]+90
                                    self.actions[idx][3] = self.actions[idx][3] - 90
                                self._env.SetAction(idx, self.actions[idx].reshape(1,-1))
            for uuvv in range(self.num_agent):
                if (int(_state[uuvv, -1]) <0):
                    print('aaa')
                    print('id:',uuvv)
                    print('action:', self.use_sp_cmd[uuvv])
                    print('now state:',self.state[uuvv,0:3])
                    print('real action:', self.actions[uuvv])
            #if (int(_state[0, -1]) > 0):
            #    break;
        self.state[:, 0] = _state[:, 0]*1
        self.state[:, 1] = _state[:, 1]*1
        state_theta_tmp = _state[:, 2].copy()
        state_theta_tmp = state_theta_tmp + 360.0
        state_theta_tmp = state_theta_tmp % 360.0
        self.state[:, 2] = state_theta_tmp.copy()
        #self.state[:, 2] = _state[:, 2]*1
        self.state[:, 3] = _state[:, 3]*1
        dx_final = (self.state[:,4]//500) - (self.state[:,0]//500)
        dy_final = (self.state[:,5]//500) - (self.state[:,1]//500)
        dis_final = abs(dx_final) + abs(dy_final)
        theta_0_360 = _state[:, 2] + 360
        theta_0_360 = theta_0_360 % 360
        X = [[self.state[i, 0] / 1000.0, self.state[i, 1] / 1000.0] for i in range(self.num_agent)]
        goal = [[self.state[i, 4] / 1000.0, self.state[i, 5] / 1000.0] for i in range(self.num_agent)]
        V = [[self.state[i, 3] / 1000.0 * math.cos(self.state[i, 2] * Pi / 180.0),
              self.state[i, 3] / 1000.0 * math.sin(self.state[i, 2] * Pi / 180.0)] for i in range(self.num_agent)]
        V_Theta = [theta_0_360[i] for i in range(self.num_agent)]

        a1 = np.cos(self.state[:, 2] / 180.0 * Pi)
        a2 = np.sin(self.state[:, 2] / 180.0 * Pi)
        b1 = (self.state[:, 4] - self.state[:, 0]) / 1000.0
        b2 = (self.state[:, 5] - self.state[:, 1]) / 1000
        c1 = (a1 * b1 + a2 * b2) / np.sqrt(b1 * b1 + b2 * b2)
        d1 = np.arccos(c1)
        # radius
        multi_ab = np.zeros((self.num_agent, LASER_NUM, DEGREE_NUM, 2), dtype='float32')
        init_xy = np.zeros((self.num_agent, LASER_NUM, DEGREE_NUM, 2), dtype='float32')
        # degree_theta = np.zeros((self.num_agent, DEGREE_NUM), dtype='float32')
        # for ii in range(self.num_agent):
        #    degree_theta[ii,:] = _state[ii, 2]*1
        # multi_ab_1 = np.zeros((self.num_agent, LASER_NUM, DEGREE_NUM), dtype='float32')
        # start_time = time.time()

        degree_real = theta_0_360.reshape(20, 1) + self.all_laser_degree
        co = np.cos(degree_real * Pi / 180.0)
        si = np.sin(degree_real * Pi / 180.0)
        for ii in range(self.num_agent):
            multi_ab[ii, :, :, 0] = self.all_laser_radius[ii, :].reshape(LASER_NUM, 1) * co[ii, :].reshape(1,DEGREE_NUM)
            multi_ab[ii, :, :, 1] = self.all_laser_radius[ii, :].reshape(LASER_NUM, 1) * si[ii, :].reshape(1,DEGREE_NUM)
            init_xy[ii, :, :, 0] = _state[ii, 0] / 1000.0
            init_xy[ii, :, :, 1] = _state[ii, 1] / 1000.0
        multi_ab = multi_ab + init_xy
        obtacles_infor = np.zeros((self.num_agent, self.num_agent + 201, LASER_SIZE, 2), dtype='float32')
        _state_do = np.concatenate((_state[:, 0:2].copy(), self.obstacle_np), axis=0)
        states_car = _state_do.reshape(self.num_agent + 201, 1, 2)
        # if 1 : #for static obtacles
        obstacle_np = abs(
            self.obstacle_np.copy() / 1000.0 - _state_do[:, 0:2].reshape(self.num_agent + 201, 1, 2) / 1000.0)
        obstacle_np_1 = obstacle_np[:, :, 0] + obstacle_np[:, :, 1]
        for ii in range(self.num_agent):
            obtacles_infor[ii, :, :, :] = multi_ab.reshape(self.num_agent, LASER_SIZE, 2)[ii, :, :].reshape(1,LASER_SIZE,2) - _state_do[:,0:2].reshape(self.num_agent + 201, 1, 2) / 1000.0
            # obs_dis = obstacle_np_1[ii,:].copy()
            # obs_dis_201 = obstacle_np_1[ii,:]<=4

            # obs_index = np.argwhere(obs_dis_201 == True)
            # multi_ab[ii,:,:,:].reshape(LASER_SIZE,2)-
            tmp_o = obtacles_infor[ii, :, :, :].copy()
            aa = tmp_o.shape[0] * tmp_o.shape[1]
            tmp_o_0 = tmp_o.reshape(aa, 2)
            tmp_1 = tmp_o_0[:, 0]
            tmp_2 = tmp_o_0[:, 1]
            i_mask = (tmp_1 <= 0.25) & (tmp_1 >= -0.25) & (tmp_2 <= 0.25) & (tmp_2 >= -0.25)
            i_index = np.argwhere(i_mask == True)
            if i_index.size > 0:
                for nn in i_index:
                    index_mask = nn % LASER_SIZE
                    self.laser_mask[ii, index_mask, 0] = 1
        for idx in range(self.num_agent):
            dis1 = self._calEuclideanDistance([self.state[idx, 4] / 1000.0, self.state[idx, 5] / 1000.0],[self.state[idx, 0] / 1000.0, self.state[idx, 1] / 1000.0])
            if self.use_sp_cmd[idx] == 0 or self.use_sp_cmd[idx] ==1:
                endpoint.append(idx)

                #return self.state,endpoint
            elif self.use_sp_cmd[idx] == 2 and _state[idx, -1] > 0:
                endpoint.append(idx)
                #return self.state,endpoint

                # self.state = _state.copy()

                #return self.state, endpoint

                #return self.state,endpoint
            #dx1 = (self.state[:, 4] // 500) - (self.state[:, 0] // 500)
            #dy1 = (self.state[:, 5] // 500) - (self.state[:, 1] // 500)


        #for iii in range(self.num_agent):
            self.reward[idx] = 0
            if 1:
                if abs(dis1-dis0[idx])<=0.02 or ((dis1-dis0[idx])>0.02):
                    self.reward[idx] =-0.1# -0.05
                elif  dis1<dis0[idx] and abs(dis1-dis0[idx])>0.02:
                    self.reward[idx] =-0.01 #-0.15
                else:
                    print('error')
            self.reward[idx] = self.reward[idx] - dis1 * 0.05
            if 0:#idx==2:
                print('d0:',d0[idx])
                print('d1:', d1[idx])
                print('v theta:',_state[idx, 2])
                print('jiajiao theta:',theta)
                print('dis0:',dis0[idx])
                print('dis1:', dis1)
            #if self.crash[i] > 0:
             #   self.reward[i] = self.reward[i] - 5.0
            #if idx == 0:
             #   print('debug')

            if 0:
                if self.use_sp_cmd[idx] == 0:
                    self.reward[idx] = self.reward[idx] - 0.1
                #if self.use_sp_cmd[idx] == 2 or self.use_sp_cmd[idx] == 3:
                #    self.reward[idx] = self.reward[idx] - 0.2
                if self.use_sp_cmd[idx] == 2 or self.use_sp_cmd[idx] == 3 or self.use_sp_cmd[idx] == 4:
                    self.reward[idx] = self.reward[idx] - 0.05
            #if min_d_agv[idx]>=0 and min_d_agv[idx]<1.4:#safe_distance
            #    self.reward[idx] = -0.1 + 0.05*min_d_agv[idx]
            if 1:
                if self.crash[idx] >0:
                    self.reward[idx] = self.reward[idx] - 2.0
            if dis_final[idx]==1:#0.2< dis1 and dis1<0.5:
                self.reward[idx] = self.reward[idx] + 50.0
                self.real_done[idx] = 1.0

            bord_flag = self._beyond_border([self.state[idx,0],self.state[idx,1]])
            if 0:#idx==0:#v_suit_flag[idx] ==False:#0:#self.tcmd[idx]>50:
                print('collid')

                render_img = self._env.Render(0)
                # aaa = cv2.resize(render*1,(500,500))
                lm = render_img.reshape(N * 50, N * 50, 3) * 1
                cv2.circle(lm, (int(_state[0][0] / 10), int(_state[0][1] / 10)), 10, (255, 0, 0), -1)
                # if 1:
                # for i_a in range(self.num_agent):
                i_a = idx*1
                for ii in range(LASER_NUM):
                    for jj in range(DEGREE_NUM):
                        if self.laser_mask[i_a][DEGREE_NUM*ii+jj][0] ==1:
                            cv2.circle(lm, (int(multi_ab[i_a][ii][jj][0] * 100), int(multi_ab[i_a][ii][jj][1] * 100)), 10,(0, 0, 0), -1)
                        else:
                            cv2.circle(lm, (int(multi_ab[i_a][ii][jj][0] * 100), int(multi_ab[i_a][ii][jj][1] * 100)), 10, (0, 0, 255), -1)
                for i_a in range(self.num_agent):
                    cv2.putText(lm, str(i_a), (int(_state[i_a, 0] / 10) - 20, int(_state[i_a, 1] / 10) + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 6)
                    cv2.putText(lm, str(i_a), (int(self.state[i_a, 4] / 10) - 20, int(self.state[i_a, 5] / 10) + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 6)
                # lm[yy:(yy + 20), xx:(xx + 20), 1] = 0
                # lm[yy:(yy + 20), xx:(xx + 20), 2] = 1
                lm = cv2.resize(lm, (500, 500))
                cv2.imwrite('/hik/home/xinchao5/agv/RVO_Py_MAS/agv_obtacles/sim_xin_ok/libenv/data/aa.png', lm)
            if bord_flag:
                self.done[idx] = 1.0
            if self.tcmd[idx] > 200:
                self.done[idx] = 1.0
        #self.endpoint = endpoint * 1
        return self.state*1, endpoint,self.reward,self.done,self.real_done,self.actions*1,self.laser_mask*1,multi_ab


    def render(self):
        self.show = True
        #if self.show: cv2.namedWindow(self.winname, 0)

    def _pre_compute_theta(self,theta_now):
        nn = theta_now / 90.0
        # nn = -1.4
        uu = math.floor(nn)
        vv = math.floor(nn + 1)
        if (nn - uu) > (vv - nn):
            num_theta = vv
        elif (nn - uu) < (vv - nn):
            num_theta = uu
        else:
            print('error')
        return num_theta
    def _pre_compute_theta_4(self,theta_now):
        nn = theta_now / 90.0
        # nn = -1.4
        uu = math.floor(nn)
        vv = math.floor(nn + 1)
        if (nn - uu) > (vv - nn):
            num_theta = vv
        elif (nn - uu) < (vv - nn):
            num_theta = uu
        else:
            print('error')
        if num_theta == -4 or num_theta == 0 or num_theta == 4:
            num_theta =0
            #self.actions[agent_idx] = [cur_x + p, 0, 0, cur_theta]

        if num_theta == -3 or num_theta == 1:
            # p = 8000
            num_theta = 1


        if num_theta == -2 or num_theta == 2:
            num_theta = 2

        if num_theta == -1 or num_theta == 3:
            num_theta = 3

        return num_theta
    def _calEuclideanDistance(self,vec1, vec2):

        mm = (vec1[0] - vec2[0]) * (vec1[0] - vec2[0]) + (vec1[1] - vec2[1]) * (vec1[1] - vec2[1])
        dist=math.pow(mm, 0.5)
        return dist

    def _beyond_border(self,vec1):
        if vec1[0]<=2000 or vec1[0]>=18000 or vec1[1]<=2000 or vec1[1]>=18000:
            flag = 1
        else:
            flag =0
        return flag

    def distance(self,pose1, pose2):
        """ compute Euclidean distance for 2D """
        return math.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2) + 0.001
    def _distance_future(self,f0, f1):
        dist = []
        for i in range(30):
            mm = self._calEuclideanDistance([f0[i, 0] / 1000.0, f0[i, 1] / 1000.0],[f1[i, 0] / 1000.0, f1[i, 1] / 1000.0])
            dist.append(mm)
        return dist
    def _fill_state(self,v1):
        #flag = 0
        v_out = np.zeros((30, 2),dtype='float32')
        try:
            first_1_num = np.where(v1[:,2]==1)[0][0]
        except:
            first_1_num = 40
            print('error')
        #np.where(v1[:, 2] == 1)[0][0]
        for aa in range(30):
            if aa<first_1_num:
                v_out[aa, 0] = v1[aa,0] * 1
                v_out[aa, 1] = v1[aa, 1] * 1
            else:
                v_out[aa, 0] = v1[first_1_num, 0] * 1
                v_out[aa, 1] = v1[first_1_num, 1] * 1
        return v_out

    def circle_laser(self,i,state_x,state_y,state_theta, radius):
        s_x = state_x.copy()
        s_y = state_y.copy()
        s_theta = state_theta.copy()
        laser_size = int((SCAN_DEGREE / EVERY_DEGREE + 1) * radius / EVERY_STEP)
        mm = np.zeros((laser_size, 2), dtype='float32')
        nn = np.zeros((laser_size, 1), dtype='float32')
        uu = 0
        fix_d = radius / float(SCAN_DEGREE / EVERY_DEGREE)
        dgree_num = int(SCAN_DEGREE / EVERY_DEGREE) + 1
        for i_r, rr in enumerate(range(1, int(radius / EVERY_STEP) + 1)):
            for i_dd, dd in enumerate(range(dgree_num)):
                #if i_dd ==12:
                #    print('ss')
                d_t = s_theta[i] - 90 + i_dd * EVERY_DEGREE
                uu = i_r * dgree_num + i_dd
                theta = d_t * Pi_pre
                mm[uu, 0] = s_x[i]/1000 + rr * fix_d * math.cos(theta)
                mm[uu, 1] = s_y[i]/1000 + rr * fix_d * math.sin(theta)
                #sss = np.where(((mm[uu, 0]-HALF_AGV_RADIUS)<=s_x[:]/1000<=(mm[uu, 0]+HALF_AGV_RADIUS)))# and (s_x[:]/1000>=(mm[uu, 0]-HALF_AGV_RADIUS)) and (s_y[:]/1000<=(mm[uu, 1]+HALF_AGV_RADIUS)) \
                    #and (s_y[:] / 1000 >= (mm[uu, 0] - HALF_AGV_RADIUS)))
                #sss = np.where(((mm[uu, 0] - HALF_AGV_RADIUS) <= s_x[:] / 1000 <= (mm[uu, 0] + HALF_AGV_RADIUS)))

                sss = (s_x<=(250+mm[uu, 0]*1000)) & (s_x>=(-250+mm[uu, 0]*1000)) & (s_y<=(250+mm[uu, 1]*1000)) & (s_y>=(-250+mm[uu, 1]*1000))
                if True in sss:
                    nn[uu, 0] = 1
                #sss=np.where(s_x<=(500+mm[uu, 0]*1000))
                #sss = np.where(s_x <= (500 + mm[uu, 0] * 1000))
                #sss = np.where(s_x <= (500 + mm[uu, 0] * 1000))
                #sss = np.where(s_x <= (500 + mm[uu, 0] * 1000))
                if 0:
                    for ii in range(self.num_agent):
                        if i!=ii:
                            if (mm[uu, 0]>=(s_x[ii]/1000-HALF_AGV_RADIUS) and mm[uu, 0]<=(s_x[ii]/1000+HALF_AGV_RADIUS)) and (mm[uu, 1]>=(s_y[ii]/1000-HALF_AGV_RADIUS) and mm[uu, 1]<=(s_y[ii]/1000+HALF_AGV_RADIUS)):
                                nn[uu, 0] = 1

                #plt.plot(mm[i_r * 12 + i_dd, 0], mm[i_r * 12 + i_dd, 1], '*', color="black")
        if 0:
            uu = mm.reshape(1,laser_size,2)
            vv =uu-self.obstacle_np
            ww = np.array((201,laser_size,1),dtype='float32')
            ww[:,:,0] = abs(vv[:,:,0])+abs(vv[:,:,1])
        return mm,nn
    def _collid_save(self,i, ss, ss_future,cs_num,ed_point):
        distance_v = []

        for i_agv in range(self.num_agent):

            v1 = [ss[i_agv, 0] / 1000.0, ss[i_agv, 1] / 1000.0]
            v2 = [ss[i, 0] / 1000.0, ss[i, 1] / 1000.0]
            d_s = self._calEuclideanDistance(v1, v2)
            distance_v.append(d_s)
            #print('aa')
        index_i = np.where(np.array(distance_v) < 5)
        ok_action = [mmm for mmm in range(5)]
        if len(index_i[0])==1:
            return ok_action
        #index_i.remove

        future_endpoint = []
        this_30 = np.zeros((30, 2), dtype='float32')
        others_30 = np.zeros((30, 2), dtype='float32')
        if i in ed_point:
            ok_action_tmp = []
            for action_i in range(5):
                self.take_future_action(i, action_i)
                #future_this = self._env.Dryrun(i, self.future_actions[i].reshape(1,-1), 30)
                future_this = self.get_single_future_state(i, action_i, self.future_actions[i])
                this_30 = future_this[0:30,0:2]*1
                #this_30 = self._fill_state(future_this)
                ok_agv_1 = []
                for other_agv_i  in index_i[0]:
                    if i!= other_agv_i:
                        future_others = self.future_states_300[other_agv_i]
                        #others_30 = self._fill_state(future_others)
                        if (cs_num[other_agv_i]+30) <= 300:
                            others_30 = future_others[cs_num[other_agv_i]:(cs_num[other_agv_i]+30),0:2]*1 #???
                        else:
                            print('others error')
                        dist_future = self._distance_future(this_30, others_30)
                        if len(np.where(np.array(dist_future) < 1)[0])<1:
                            ok_agv_1.append(other_agv_i)
                        else:
                            uuy = 1
                            if 0:
                                print('collid')

                                render_img = self._env.Render(0)
                                # aaa = cv2.resize(render*1,(500,500))
                                lm = render_img.reshape(N * 50, N * 50, 3)*1
                                yy = int(self.state[i,1] / 10)
                                xx = int(self.state[i,0] / 10)
                                lm[yy:(yy+20),xx:(xx+20),0] = 0

                                yy = int(self.state[other_agv_i, 1] / 10)
                                xx = int(self.state[other_agv_i, 0] / 10)
                                lm[yy:(yy + 20), xx:(xx + 20), 1] = 0
                                #lm[yy:(yy + 20), xx:(xx + 20), 1] = 0
                                #lm[yy:(yy + 20), xx:(xx + 20), 2] = 1
                                lm = cv2.resize(lm, (500, 500))
                                cv2.imwrite('/hik/home/xinchao5/agv/RVO_Py_MAS/agv_obtacles/sim_xin_ok/libenv/data/aa.png',lm)
                if (len(index_i[0])-1) == len(ok_agv_1):
                    ok_action_tmp.append(action_i)
            ok_action = ok_action_tmp*1
        else:
            print('error error')
            #future_endpoint

        return ok_action

if __name__ == "__main__":
    random.seed('hello')

    num_agent = 1
    menv = MyENV(num_agent)
    start_num = len(menv.start_port)
    target_num = len(menv.target_port)
    actions = [1,1,3,2,1]
    last_action = [0]*num_agent
    all_actions = []
    for _ in range(100):
        state, endpoint = menv.reset()
        menv.render()
        for iii in range(1000):
            for ii in range(num_agent):
                if  ii in endpoint:
                    for idx in range(num_agent):
                        act = random.randint(0, 3)
                        all_actions.append(act)
                        menv.set_action(idx, act)
                        last_action = act

            states,endpoint = menv.step()
            print('last_action',last_action,'states:',states)
            #print(np.floor(state[0, [0, 1, 2, 3, 6, 7]]).tolist(), reward[0], menv.total_reward[0], endpoint)
            #print(np.floor(state[1, [0, 1, 2, 3, 6, 7]]).tolist(), reward[1], menv.total_reward[1], endpoint)
            # for i, t in enumerate(done):
    print('done')

