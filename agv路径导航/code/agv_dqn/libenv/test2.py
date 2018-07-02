import cv2
Debug = 0
WaitTime = 1
Debug_Info = 1
img = cv2.imread('../data/out_40.png', -1)
if Debug:
    cv2.namedWindow('render',0)

import example
import numpy as np
import random
import math
import matplotlib.pyplot as plt
TIME = 30
A = 1000.0
T = 0.5
aa = plt.subplot(1, 1, 1)
import cmd2pvt
def pre_compute_theta(theta_now):
    nn = theta_now/90.0
    #nn = -1.4
    uu = math.floor(nn)
    vv= math.floor(nn+1)
    if (nn-uu)>(vv-nn):
        num_theta = vv
    elif (nn-uu)<(vv-nn):
        num_theta = uu
    else:
        print('error')
    return num_theta

def run(show,ii,action_i,ss):

    cur_x= ss[0][0]
    cur_y = ss[0][1]
    cur_theta = ss[0][2]
    cur_v = ss[0][3]
    render = mm.Render(Debug_Info)
    theta_num = pre_compute_theta(cur_theta)
    if action_i == 1:
        a=A
        p=2000

        if theta_num == -4 or theta_num == 0 or theta_num == 4:
            action = [cur_x + p, 0, 0, cur_theta]


        if theta_num == -3 or theta_num == 1:

            # p = 8000
            action = [cur_y + p, 0, 0, cur_theta]

        if theta_num == -2 or theta_num == 2:
            action = [cur_x - p, 0, 0, cur_theta]

        if theta_num == -1 or theta_num == 3:
            # p = 8000
            action = [cur_y - p, 0, 0, cur_theta]
        mm.SetAction(0, np.array([action], dtype=np.float32))
        for i in range(TIME):
            states = mm.Step()
            a1 = aa.plot(ii + i, states[0][3], 'ro--', label='line 1')
            # a1 = a.plot(time_t*mmmm+i, states[0][3], 'ro--', label='line 1')
            # b1 = b.plot(time_t*mmmm+i, aa, 'ro--', label='line 1')
            # render = mm.Render(0)
            if show:
                cv2.namedWindow('render', 0)
                cv2.imshow('render', render.reshape(50 * 40, 50 * 40, 3))
                cv2.waitKey(WaitTime);
        ii = ii + i
    if action_i ==0:
        a=A
        #next_v = cur_v + a * T
        p = cur_v*cur_v/(2*a)
        if theta_num == -4 or theta_num == 0 or theta_num == 4:
            action = [cur_x + p, 0, 0, cur_theta]


        if theta_num == -3 or theta_num == 1:

            # p = 8000
            action = [cur_y + p, 0, 0, cur_theta]

        if theta_num == -2 or theta_num == 2:
            action = [cur_x - p, 0, 0, cur_theta]

        if theta_num == -1 or theta_num == 3:
            # p = 8000
            action = [cur_y - p, 0, 0, cur_theta]
        #action = [p, next_v, 200, cur_theta]
        mm.SetAction(0, np.array([action], dtype=np.float32))
        for i in range(TIME):
            states = mm.Step()
            a1 = aa.plot(ii + i, states[0][3], 'ro--', label='line 1')
            # a1 = a.plot(time_t*mmmm+i, states[0][3], 'ro--', label='line 1')
            # b1 = b.plot(time_t*mmmm+i, aa, 'ro--', label='line 1')
            # render = mm.Render(0)
            if show:
                cv2.namedWindow('render', 0)
                cv2.imshow('render', render.reshape(50 * 40, 50 * 40, 3))
                cv2.waitKey(WaitTime);
        ii = ii + i
    #action = [cur_x + p, 0, 0, cur_theta]

    if action_i == 2:
        if theta_num == -4 or theta_num == 0 or theta_num == 4:
            a = -1 * A
            next_v = 0
            p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
            # p = 8000
            action = [cur_x + p, 0, 0, cur_theta]

        if theta_num == -3 or theta_num == 1:
            a = -1 * A
            next_v = 0
            p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
            # p = 8000
            action = [cur_y + p, 0, 0, cur_theta]

        if theta_num == -2 or theta_num == 2:
            a = -1 * A
            next_v = 0
            p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
            # p = 8000
            action = [cur_x - p, 0, 0, cur_theta]

        if theta_num == -1 or theta_num == 3:
            a = -1 * A
            next_v = 0
            p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
            # p = 8000
            action = [cur_y - p, 0, 0, cur_theta]

        mm.SetAction(0, np.array([action], dtype=np.float32))
        for i in range(2000):
            states = mm.Step()
            a1 = aa.plot(ii + i, states[0][3], 'ro--', label='line 1')
            # a1 = a.plot(time_t*mmmm+i, states[0][3], 'ro--', label='line 1')
            # b1 = b.plot(time_t*mmmm+i, aa, 'ro--', label='line 1')
            # render = mm.Render(0)
            if show:
                cv2.namedWindow('render', 0)
                cv2.imshow('render', render.reshape(50 * 40, 50 * 40, 3))
                cv2.waitKey(WaitTime);
            if states[:, -1]> 0:
                ii = ii + i
                break
    if action_i == 3:


        if theta_num == -4 or theta_num == 0 or theta_num == 4:
            a = -1 * A
            next_v = 0
            p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
            p_p = cur_x + p
            # p = 8000
            #action = [cur_x + p, 0, 0, theta_num*90+90]

        if theta_num == -3 or theta_num == 1:
            a = -1 * A
            next_v = 0
            p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
            # p = 8000
            p_p = cur_y + p
            #action = [cur_y + p, 0, 0, theta_num*90+90]

        if theta_num == -2 or theta_num == 2:
            a = -1 * A
            next_v = 0
            p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
            p_p = cur_x - p
            # p = 8000
            #action = [cur_x - p, 0, 0, theta_num*90+90]

        if theta_num == -1 or theta_num == 3:
            a = -1 * A
            next_v = 0
            p = abs(next_v * next_v - cur_v * cur_v) / (2 * A)
            p_p = cur_y - p
            # p = 8000
            #action = [cur_y - p, 0, 0, theta_num*90+90]
            #action = [cur_y - p, 0, 0, theta_num * 90 + 90]
        action = [p_p, 0, 0, theta_num * 90]
        mm.SetAction(0, np.array([action], dtype=np.float32))
        for i in range(2000):
            states = mm.Step()
            a1 = aa.plot(ii + i, states[0][3], 'ro--', label='line 1')
            # a1 = a.plot(time_t*mmmm+i, states[0][3], 'ro--', label='line 1')
            # b1 = b.plot(time_t*mmmm+i, aa, 'ro--', label='line 1')
            # render = mm.Render(0)
            if show:
                cv2.namedWindow('render', 0)
                cv2.imshow('render', render.reshape(50 * 40, 50 * 40, 3))
                cv2.waitKey(WaitTime);
            if states[:, -1]> 0:
                ii = ii + i
                break
        action = [p_p, 0, 0, theta_num*90+90]
        mm.SetAction(0, np.array([action], dtype=np.float32))
        for i in range(2000):
            states = mm.Step()
            a1 = aa.plot(ii + i, states[0][3], 'ro--', label='line 1')
            # a1 = a.plot(time_t*mmmm+i, states[0][3], 'ro--', label='line 1')
            # b1 = b.plot(time_t*mmmm+i, aa, 'ro--', label='line 1')
            # render = mm.Render(0)
            if show:
                cv2.namedWindow('render', 0)
                cv2.imshow('render', render.reshape(50 * 40, 50 * 40, 3))
                cv2.waitKey(WaitTime);
            if states[:, -1] > 0:
                ii = ii + i
                break


    if 1:
        plt.title("My matplotlib learning")
        plt.xlabel("X")
        plt.ylabel("Y")
        handles, labels = aa.get_legend_handles_labels()
        aa.legend(handles[::-1], labels[::-1])
        plt.savefig('plot_p.png')
    return ii, states

        #print(states[:,-1])


if 1:
    mm = example.Env(img, 1)
    #print(mm.GetStartPort())
    #print(mm.GetEndPort())
    #print(mm.GetObsPort())
    ss = mm.Reset([6], [3,12])
    start_grid = int(ss[0][0]/500)
    print(start_grid)
    qq = pre_compute_theta(-179.97399902)

    #action = [500*13+250, 0, 0, 0]
    #mm.SetAction(0, np.array([action], dtype=np.float32))
    #run()

    #action = [500*3+250, 0, 0, 180]
    #mm.SetAction(0, np.array([action], dtype=np.float32))
    #pvt = cmd2pvt.acce2pvt(1, v0)

    actions = [1,1,3,1,1,0,0,0,0,0,0,0,0]
    print(ss)
    all_theta = []
    global_t = 0
    all_action = []
    for index_i,action in enumerate(actions):
    #if index_i==27:
            #print('stop')
    #for index_i in range(100):
        #action = random.randint(0,3)
        print('take_action---:',action)
        all_action.append(action)
        global_t, ss = run(False, global_t, action,ss)
        all_theta.append(ss[0][2])
        print('----------------------------')
        print('action:',action,'ss:',ss)
        if ss[0][-1]<0:
            print(all_action)
            print('error')
    all_theta.sort()
    print(all_theta)
    print('done')
    print(ss)
