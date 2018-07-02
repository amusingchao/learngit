import numpy as np
import math
import cv2
import cmd2pvt_random
import matplotlib.pyplot as plt
import random
STATE_DIM = 7
STEP_T = 1#0.02
X_NUM = 0
Y_NUM  = 1
V_NUM = 2
V_THETA_NUM = 3
R_NUM = 4
Gx_NUM = 5
Gy_NUM = 6


STATE_X = 0
STATE_Y = 1
STATE_THETA = 2
STATE_VEL = 3
STATE_CRASH_INFO = 4
STATE_CMD_CODE = 5
Pi = 3.1415926
ROBO_RADIUS = 0.5
A0 = 0 #+
A1 = 1 #-
A2 = 2 #rotate left 90
A3 = 3 #rotate right -90
#A4 = 4 #rotate right 180
T_e = 0.2
A_e = 1000.0 #1000.0
START_NUM = 4
END_NUM = 12
class AGV():
    def __init__(self,mm,show,work_num):
        self.mm = mm
        #self.target_p=[]
        start_num = [2,3,6,7]
        start_index = random.randint(0, START_NUM-1)
        self.start_num = 6#start_num[start_index]#random.randint(0, START_NUM)
        final_num = [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17]
        final_index = random.randint(0, END_NUM - 1)
        self.end_num = final_num[final_index] #10

        self.target_p = self.mm.GetEndPort()[self.end_num]
        #self.target_p.append(8750.0)
        #self.target_p.append(6750.0)

        self.show = show
        #self.a_draw = plt.subplot(1, 1, 1)
        #print('aa')
    def reset(self):
        s=self.mm.Reset([self.start_num,7], [self.end_num])
        #act = [1750, 0, 0, 0]
        #self.mm.SetAction(0, np.array([act], dtype=np.float32))
        #s, collided_num0 = self._run()
        state_np = np.zeros(STATE_DIM, np.float64)
        state_np[X_NUM] = s[0][STATE_X]/1000.0
        state_np[Y_NUM] = s[0][STATE_Y]/1000.0
        state_np[V_NUM] = s[0][STATE_VEL]/1000.0
        state_np[V_THETA_NUM] = s[0][STATE_THETA] * Pi / 180.0
        state_np[R_NUM] = ROBO_RADIUS
        state_np[Gx_NUM] = self.target_p[0]/1000.0
        state_np[Gy_NUM] = self.target_p[1]/1000.0
        self.reward = 0
        self.collided = False
        self.terminal = False
        return state_np,s
        #return [self.start_p,self.start_v,self.radius,self.taget_p]


    def step(self,current_state,exe_action,agv_states):

        v0 = agv_states[0][STATE_VEL]
        theta_0 = agv_states[0][STATE_THETA]
        if agv_states[0][STATE_CMD_CODE] == 1:
            action1 = [-1, 0, 0, theta_0];
            action1 = np.array([action1], dtype=np.float32)
            self.mm.SetAction(0, action1)
            states1, collided_num1 = self._run()
        collided_num = 0
        collided = False
        rotate_flag = 0
        try:
            p, theta = cmd2pvt_random.acce2pvt(exe_action, agv_states[0][STATE_X], agv_states[0][STATE_Y], agv_states[0][STATE_THETA])
        except:
            p, theta = cmd2pvt_random.acce2pvt(exe_action, agv_states[0][STATE_X], agv_states[0][STATE_Y],\
                                        agv_states[0][STATE_THETA])
            print('aa')
        if abs(theta - 90) < 2:
            theta = 90
        if abs(theta - 180) < 2:
            theta = 180
        if abs(theta + 90) < 2:
            theta = -90
        if abs(theta) < 2:
            theta = 0
        if abs(theta + 180) < 2:
            theta = 180
        #for rotate
        act = [p, 0, 0, theta]
        self.mm.SetAction(0, np.array([act], dtype=np.float32))
        states,collided_num0= self._run()

        # for move
        act = [p, 0, 0, theta]
        self.mm.SetAction(0, np.array([act], dtype=np.float32))
        states, collided_num1 = self._run()

        #if exe_action == 30 :
        rotate_flag = 0
        #print('real action----', action, ' ',states[0][STATE_CMD_CODE])


        next_state = np.zeros(STATE_DIM,np.float64)
        next_state[X_NUM] = states[0][STATE_X]/1000.0
        next_state[Y_NUM] = states[0][STATE_Y]/1000.0
        next_state[V_NUM] = states[0][STATE_VEL]/1000.0
        next_state[V_THETA_NUM] = states[0][STATE_THETA] * Pi / 180.0
        next_state[R_NUM] = ROBO_RADIUS
        next_state[Gx_NUM] = self.target_p[0] /1000.0
        next_state[Gy_NUM] = self.target_p[1]/1000.0

        collided = 0
        robo_radius = ROBO_RADIUS
        terminal = self._terminal(next_state[X_NUM], next_state[Y_NUM], robo_radius, self.target_p,states[0][STATE_CMD_CODE])
        #reward = self._reward(terminal, collided,p_distance)
        #vx = math.cos(current_state[V_THETA_NUM])
        #vy = math.sin(current_state[V_THETA_NUM])
        vx =next_state[X_NUM]-current_state[X_NUM] #math.cos(current_state[V_THETA_NUM])
        vy =next_state[Y_NUM]-current_state[Y_NUM] #math.sin(current_state[V_THETA_NUM])
        reward = self._reward(terminal, collided,rotate_flag,current_state,next_state,self.target_p)
        return next_state, reward, terminal, states


    def _run(self):
        states = None
        col_num = 0
        for i in range(10000):
            states = self.mm.Step()
            if states[0][STATE_CRASH_INFO] > 0:
                col_num = col_num + 1
            render = self.mm.Render(0)
            if self.show ==True:
                #a = plt.subplot(1, 1, 1)


                print(render.shape)
                cv2.namedWindow('render',0)
                cv2.imshow('render', render.reshape(2000,2000,3))
                cv2.waitKey(1);
                #a1 = self.a_draw.plot(i, states[0][3], 'ro--', label='line 1')
            if states[0, -1] > 0: break;
        return states,col_num
    def _terminal(self,next_x, next_y,r_agv, target_position,states_v):
        terminal_result = False
        result_distance = math.sqrt(math.pow(next_x - (target_position[0]/1000.0), 2) + math.pow(next_y - (target_position[1]/1000.0), 2))
        if result_distance <= (0.2 + 0.1):
            terminal_result = 1
        return terminal_result


    def _reward(self, terminal, collided, r_flag,cur_p,next_p,target_p):

        if terminal == 1:
            return 100.0

        if r_flag == 0:
            cur_target_distance = math.sqrt(math.pow(cur_p[X_NUM] - (target_p[0]/1000.0), 2) + math.pow(cur_p[Y_NUM] - (target_p[1]/1000.0), 2))
            next_target_distance = math.sqrt(math.pow(next_p[X_NUM] - (target_p[0] / 1000.0), 2) + math.pow(next_p[Y_NUM] - (target_p[1] / 1000.0), 2))
            if next_target_distance<cur_target_distance:
                return -1.0
            else:
                return -10.0
        if r_flag == 1:
            if abs(cur_p[V_THETA_NUM])<0.1 or abs(cur_p[V_THETA_NUM]-Pi)<0.1 or abs(cur_p[V_THETA_NUM]+Pi)<0.1:
                cur_vel_degree =[1.0,0]
            if abs(cur_p[V_THETA_NUM]-Pi/2.0) < 0.1 or abs(cur_p[V_THETA_NUM] + Pi/2.0) < 0.1:
                cur_vel_degree = [0, 1.0]
            if abs(next_p[V_THETA_NUM])<0.1 or abs(next_p[V_THETA_NUM]-Pi)<0.1 or abs(next_p[V_THETA_NUM]+Pi)<0.1:
                next_vel_degree =[1.0,0]
            if abs(next_p[V_THETA_NUM]-Pi/2.0) < 0.1 or abs(next_p[V_THETA_NUM] + Pi/2.0) < 0.1:
                next_vel_degree = [0, 1.0]
            vec_cur_distance = [(target_p[0]/1000.0-cur_p[X_NUM]),(target_p[1]/1000.0-cur_p[Y_NUM])]
            vec_next_distance = [(target_p[0]/1000.0 - next_p[X_NUM]), (target_p[1]/1000.0 - next_p[Y_NUM])]
            cur_dot = vec_cur_distance[0]*cur_vel_degree[0] + vec_cur_distance[1]*cur_vel_degree[1]
            next_dot = vec_next_distance[0] * next_vel_degree[0] + vec_next_distance[1] * next_vel_degree[1]
            cur_p_degree = math.acos(abs(cur_dot)/(math.sqrt(vec_cur_distance[0]*vec_cur_distance[0]+vec_cur_distance[1]*vec_cur_distance[1])+1))
            next_p_degree = math.acos(abs(next_dot) / (math.sqrt(vec_next_distance[0] * vec_next_distance[0] + vec_next_distance[1] * vec_next_distance[1]) + 1))
            if next_p_degree<cur_p_degree:
                return -1.0
            else:
                return -10.0
        # time penalty or collision penalty



