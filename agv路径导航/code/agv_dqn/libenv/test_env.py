import numpy as np
import random
import sys
import cv2

map_img = cv2.imread('./data/out_40.png', -1)

from libenv.example import Env

###########################################

N = 40
N_A = 4
waittime = 0
debug = 0
import math
TIME = 30
A=1000.0
T=0.5

class MyENV:
    def __init__(self, num_agent, winname='render', filename='log'):

        self.observation_space_shape = [4]
        self.action_space_n = N * 2
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

        self.cmd_state = np.zeros((num_agent,), dtype='int32')
        self.actions = np.zeros((num_agent, 4), dtype='float32')

        self.mask = np.zeros((N, N), dtype='uint16')
        for x, y in self.target_port:
            self.mask[int(y / 500), int(x / 500)] = 500
        for x, y in self.obstacle_port:
            self.mask[int(y / 500), int(x / 500)] = 500

    def reset(self):

        self.t = 0
        self.show = False

        s_idx = random.sample(range(0, len(self.start_port)), self.num_agent)
        d_idx = random.sample(range(0, len(self.target_port)), self.num_agent)

        _state = self._env.Reset(s_idx, d_idx)

        self.state = np.zeros((self.num_agent, self.observation_space_shape[0]),
                              dtype='float32')  # x0, y0, x1, y1, xclean, yclean
        self.state= _state.copy()



        return self.state * 0.002, range(self.num_agent)

    def ireset(self, idx):

        self.show = False

        s_idx = random.randrange(0, len(self.start_port))
        d_idx = random.randrange(0, len(self.target_port))

        _state = self._env.iReset(idx, s_idx, d_idx)
        self.state = _state.copy()


        # self.state[idx, 48] = _state[idx, 0]
        # self.state[idx, 49] = _state[idx, 1]

        return _state

    def set_action(self, agent_idx, act):

        cur_x = self.state[0][0]
        cur_y = self.state[0][1]
        cur_theta = self.state[0][2]
        cur_v = self.state[0][3]
        #render = mm.Render(Debug_Info)
        theta_num = self._pre_compute_theta(cur_theta)
        if act == 1:
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
            a = A
            # next_v = cur_v + a * T
            p = cur_v * cur_v / (2 * a)
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
            # action = [p, next_v, 200, cur_theta]
            self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))

        # action = [cur_x + p, 0, 0, cur_theta]

        if act == 2:
            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                self.actions[agent_idx] = [cur_x + p, 0, 0, cur_theta]

            if theta_num == -3 or theta_num == 1:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                self.actions[agent_idx] = [cur_y + p, 0, 0, cur_theta]

            if theta_num == -2 or theta_num == 2:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                self.actions[agent_idx] = [cur_x - p, 0, 0, cur_theta]

            if theta_num == -1 or theta_num == 3:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                self.actions[agent_idx] = [cur_y - p, 0, 0, cur_theta]

            self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))

                # a1 = a.plot(time_t*mmmm+i, states[0][3], 'ro--', label='line 1')
                # b1 = b.plot(time_t*mmmm+i, aa, 'ro--', label='line 1')
                # render = mm.Render(0)

        if act == 3:

            if theta_num == -4 or theta_num == 0 or theta_num == 4:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x + p
                # p = 8000
                # action = [cur_x + p, 0, 0, theta_num*90+90]

            if theta_num == -3 or theta_num == 1:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                # p = 8000
                p_p = cur_y + p
                # action = [cur_y + p, 0, 0, theta_num*90+90]

            if theta_num == -2 or theta_num == 2:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_x - p
                # p = 8000
                # action = [cur_x - p, 0, 0, theta_num*90+90]

            if theta_num == -1 or theta_num == 3:
                a = -1 * A
                next_v = 0
                p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
                p_p = cur_y - p
                # p = 8000
                # action = [cur_y - p, 0, 0, theta_num*90+90]
                # action = [cur_y - p, 0, 0, theta_num * 90 + 90]
            self.actions[agent_idx] = [p_p, 0, 0, theta_num ]  #special
            self._env.SetAction(agent_idx, self.actions[agent_idx].reshape(1, -1))



    def step(self):

        # dx0 = (self.state[:,2]//500) - (self.state[:,0]//500)
        # dy0 = (self.state[:,3]//500) - (self.state[:,1]//500)
        # dis0 = abs(dx0) + abs(dy0)

        ttmp = 0


        while len(endpoint) == 0 or ttmp<=TIME:
            ttmp += 1

            _state = self._env.Step()
            # self.crash += (_state[:,-2] == 1)
            self.crash = self.crash + _state[:, -2]
            if self.show:
                render_img = self._env.Render(debug)
                lm = render_img.reshape(N * 50, N * 50, 3)
                lm = cv2.resize(lm, (500, 500))
                cv2.imshow(self.winname, lm)
                cv2.waitKey(waittime)

            for idx in range(self.num_agent):
                if _state[idx, -1] > 0:

                    if self.cmd_state[idx] > 0:
                        self.cmd_state[idx] = 0
                        endpoint.append(idx)
                    else:
                        self.cmd_state[idx] = 1
                        self._env.SetAction(idx, self.actions[idx].reshape(1, -1))

                elif _state[idx, -1] < 0:
                    endpoint.append(idx)
        self.state = _state.copy()



    def render(self):
        self.show = True
        # if self.show: cv2.namedWindow(self.winname, 0)

    def pre_compute_theta(self,theta_now):
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

    def calEuclideanDistance(vec1, vec2):
        mm = (vec1[0] - vec2[0]) * (vec1[0] - vec2[0]) + (vec1[1] - vec2[1]) * (vec1[1] - vec2[1])
        dist = math.pow(mm, 0.5)
        return dist

if __name__ == "__main__":
    random.seed('hello')

    num_agent = 20
    menv = MyENV(num_agent)
    start_num = len(menv.start_port)
    target_num = len(menv.target_port)

    for _ in range(100):
        state, endpoint = menv.reset()
        menv.render()
        for _ in range(1900):

            for idx in endpoint:
                act = random.randint(0, 3)
                menv.set_action(idx, act)

            state, reward, done, endpoint = menv.step()
            print(np.floor(state[0, [0, 1, 2, 3, 6, 7]]).tolist(), reward[0], menv.total_reward[0], endpoint)
            print(np.floor(state[1, [0, 1, 2, 3, 6, 7]]).tolist(), reward[1], menv.total_reward[1], endpoint)
            # for i, t in enumerate(done):
            for i in endpoint:
                if done[i] > 0:
                    # print(i, menv.tcmd[i], state[i,:4].tolist(), reward[i], menv.total_reward[i], endpoint)
                    # __import__('ipdb').set_trace()
                    menv.ireset(i)
