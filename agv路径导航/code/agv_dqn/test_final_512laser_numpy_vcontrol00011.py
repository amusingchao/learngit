# coding:utf-8

import os
import random
import numpy as np
import tensorflow as tf
import time
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from keras.models import Sequential
from keras.layers import Dense

# import myEnv
import env_final_512laser_numpy_vcontrol00011 as myEnv
import replay_buffer
PI = 3.1415926
ENV_NAME = 'AGV_final_512laser_numpy_vcontrol00011'
NUM_EPISODES = 1200000  # Number of episodes the agent plays
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS = 2500000#1500000#2500000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 20000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 100000  # Number of replay memory the agent uses for training
BATCH_SIZE = 128  # Mini batch size
TARGET_UPDATE_INTERVAL = 5000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 1  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 5000  # The frequency with which the network is saved
NO_OP_STEPS = 10  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
SAVE_NETWORK_PATH = 'model_pp/' + ENV_NAME
SAVE_SUMMARY_PATH = 'log_pp/'  + ENV_NAME
NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time

OBS_SIZE = []

LOAD_NETWORK = True
TRAIN = True

if 1:
    N = 40
    N_A = 4
    waittime = 1
    debug = 0
    import math

    TIME = 30
    A = 1000.0
    T = 0.5
    V_MAX = 2.0
    DEBUG_F = 0
    show = True
    SCAN_DEGREE = 180
    EVERY_DEGREE = 12
    EVERY_STEP = 0.4
    Pi = 3.1415926
    Pi_pre = Pi / 180.0
    LASER_RADIUS = 4
    AGV_RADIUS = 0.5
    HALF_AGV_RADIUS = 0.25
    DEGREE_NUM = int(SCAN_DEGREE / EVERY_DEGREE) + 1
    LASER_NUM = int(LASER_RADIUS / EVERY_STEP)
    EVERY_RADIUS = LASER_RADIUS / float(LASER_NUM)
    LASER_SIZE = int((SCAN_DEGREE / EVERY_DEGREE + 1) * LASER_RADIUS / EVERY_STEP)


class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        self.all_reward = []
        self.all_v = []
        self.all_target_v = []

        # Create replay memory
        self.replay_memory = replay_buffer.ReplayBuffer(NUM_REPLAY_MEMORY)

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.initialize_all_variables())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_network(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(OBS_SIZE[0],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None] + OBS_SIZE)
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        # optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    # def run(self, state, action, reward, terminal, observation):
    #     next_state = np.append(state[1:, :, :], observation, axis=0)
    def run(self, observation, action, reward, terminal, real_terminal, new_observation, idx=0):

        summary_str = None
        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
    ##### reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.add(observation, action, reward, new_observation, float(real_terminal),idx)

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                #print('Successfully saved: ' + save_path)

        if idx == 0:
            self.total_reward += reward
            t_q = np.max(self.q_values.eval(feed_dict={self.s: [np.float32(observation)]}))
            #t1_q = np.max(self.target_q_values.eval(feed_dict={self.st: [np.float32(new_observation)]}))
            self.total_q_max += t_q
            self.duration += 1
            self.all_reward.append(reward)
            self.all_v.append(t_q)
            #self.all_target_v.append(t1_q)
            if terminal or real_terminal:
                # Write summary
                if self.t >= INITIAL_REPLAY_SIZE:
                    stats = [self.total_reward, self.total_q_max / float(self.duration),
                            self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)),self.epsilon]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)

                # Debug
                if self.t < INITIAL_REPLAY_SIZE:
                    mode = 'random'
                elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                    mode = 'explore'
                else:
                    mode = 'exploit'
                print(
                    'done',
                    'run step:',self.t,
                    'step:', self.duration,
                    "Ep:", self.episode + 1,
                    "| Ep_r: ", self.total_reward,
                    'max_value:', self.total_q_max / float(self.duration),
                    'loss:', self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)),
                    'epison:',self.epsilon)
                #print('all reward:',self.all_reward)
                #print('all q:', self.all_v)
                #print('all_target_v:', self.all_target_v)
                self.all_reward = []
                self.all_v = []
                self.all_target_v = []
                self.total_reward = 0
                self.total_q_max = 0
                self.total_loss = 0
                self.duration = 0
                self.episode += 1

        self.t += 1

        return new_observation, summary_str

    def train_network(self):

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch \
                = self.replay_memory.sample(BATCH_SIZE)

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch))})
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch)),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + 'reward', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + 'maxq', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + 'step', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + 'loss', episode_avg_loss)

        episode_epison = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + 'epison', episode_epison)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss,episode_epison]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        # summary_op = tf.merge_all_summaries()
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        #if random.random() <= 0.05:
            #action = random.randrange(self.num_actions)
        #else:
        action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))

        self.t += 1

        return action

if 0:
    def wrapper_state(state, new_state):

        # x, y, target_x, target_y, road_x, road_y, next_x, next_y
        new_state[:,:8] = state[:,:]

        new_state[0,8:16] = state[1,:]
        new_state[1,8:16] = state[0,:]

def concat_state(state, new_state,agv_n,ls_mask):
    l_mask = ls_mask.copy()
    paixu_state = state[:, 0:2].copy()
    # dis_state = state.copy()
    # np_rvo = np.array(last_rvo_vflag).reshape(agv_n,1)
    # x, y, target_x, target_y, road_x, road_y, next_x, next_y
    ls_mask_size = l_mask.size / agv_n
    new_state[:, :6] = state[:, :]
    # new_state[:, 6:7] = np_rvo.copy()
    new_state[:, 6*4:(6*4 + ls_mask_size)] = l_mask.reshape(agv_n, ls_mask_size)
    # aa= 1
    if 1:
        all_list = [mmm for mmm in range(agv_n)]
        for al in all_list:
            dis_state = paixu_state - paixu_state[al].reshape(1, 2)
            dis_state = abs(dis_state)
            distance_state = dis_state[:, 0] + dis_state[:, 1]
            sort_index = list(np.argsort(distance_state))
            else_list = sort_index * 1  # [nnn for nnn in range(agv_n)]
            else_list.remove(al)
            # random.shuffle(else_list)
            else_list_1 = else_list[0:3]*1
            for jj, jj_list in enumerate(else_list_1):
                new_state[al, 6 * (jj + 1):6 * (jj + 2)] = state[jj_list, :]
def concat_state_x(state, new_state,agv_n,ls_mask):
    l_mask = ls_mask.copy()
    paixu_state = state[:,0:2].copy()
    #dis_state = state.copy()
   # np_rvo = np.array(last_rvo_vflag).reshape(agv_n,1)
    # x, y, target_x, target_y, road_x, road_y, next_x, next_y
    ls_mask_size = l_mask.size/agv_n
    new_state[:,:6] = state[:,:]
    #new_state[:, 6:7] = np_rvo.copy()
    new_state[:,120:(120+ls_mask_size)] = l_mask.reshape(agv_n,ls_mask_size)
    #aa= 1
    if 1:
        all_list = [mmm for mmm in range(agv_n)]
        for al in all_list:
            dis_state = paixu_state - paixu_state[al].reshape(1,2)
            dis_state = abs(dis_state)
            distance_state = dis_state[:,0]+dis_state[:,1]
            sort_index = list(np.argsort(distance_state))
            else_list = sort_index*1 #[nnn for nnn in range(agv_n)]
            else_list.remove(al)
            #random.shuffle(else_list)
            else_list_1 = else_list*1
            for jj, jj_list in enumerate(else_list_1):
                new_state[al,6*(jj+1):6*(jj+2)] = state[jj_list,:]
def wrapper_state(state):

    # x, y, target_x, target_y, road_x, road_y, next_x, next_y
    state[:, 0] = state[:, 0]/1000.0
    state[:, 1] = state[:, 1]/1000.0
    state[:, 2] = state[:, 2]/180.0*PI
    state[:, 3] = state[:, 3]/1000.0
    state[:, 4] = state[:, 4]/1000.0
    state[:, 5] = state[:, 5]/1000.0


def main():
    # mm = float(677) < 512
    random.seed('hello')
    num_agent = 20
    env = myEnv.MyENV(num_agent)

    env_action_space_n = env.action_space_n
    env_laser_size = env.laser_size
    OBS_SIZE[:] = [(i*4+env_laser_size) for i in env.observation_space_shape]
    #OBS_SIZE[:] = [i+1+env_laser_size  for i in env.observation_space_shape]
    agent = Agent(num_actions=env_action_space_n)
    #observation = np.zeros((num_agent, OBS_SIZE[0]), dtype='float32')
    #new_observation = np.zeros((num_agent, OBS_SIZE[0]), dtype='float32')
    #new_observation = np.zeros((num_agent, OBS_SIZE[0]), dtype='float32')
    #init_observation_1 = np.zeros((num_agent, OBS_SIZE[0]/num_agent), dtype='float32')
    #init_observation = np.zeros((num_agent, OBS_SIZE[0]), dtype='float32')
    buffer_observation = np.zeros((num_agent, OBS_SIZE[0]), dtype='float32')
    buffer_new_observation = np.zeros((num_agent, OBS_SIZE[0]), dtype='float32')
    if TRAIN:  # Train mode
        summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH)
        #start_time = time.time()
        observation_, endpoint,laser_mask,mask_xy_ = env.reset()
        #observation_1 = np.floor(observation_)
        #wrapper_state(observation_1, observation, num_agent)

        action = [0] * num_agent
        last_action = [0] * num_agent
        all_actions = [[] for qqq in range(num_agent)]
        all_states = [[] for rrr in range(num_agent)]
        all_p = [[] for sss in range(num_agent)]
        env.render()
        wrapper_state(observation_)

        concat_state(observation_, buffer_observation, num_agent,laser_mask)
        while True:

            for idx in endpoint:
                #init_observation_1[idx] = observation_1[idx].copy()

                #$init_observation[idx] = observation[idx].copy()
                #if idx == 12:
                #    print('print debug')
                concat_state(observation_, buffer_observation, num_agent,laser_mask)
                try:
                    #action[idx] = agent.get_action(observation_[idx])
                    action[idx] = agent.get_action_at_test(buffer_observation[idx])
                except:
                    print('error')
                #if action[idx]==3:
                  #  print('33:')
                #print('new action:',action[idx])
                #if action[idx] == 2 or action[idx] == 3:
                #    print('special')
                last_action[idx] = action[idx]*1
                all_actions[idx].append(last_action[idx] * 1)
                #all_actions[idx].append(action[idx])
                env.set_action(idx, action[idx])

            new_observation_,endpoint,reward,terminal,real_terminal,real_p,laser_mask,mask_xy = env.step()

            #if terminal[idx, 0]:
                #print('one episode')
            #print('last action:',last_action,'new_observation_:',new_observation_,'reward:',reward,terminal)



            # if agent.episode % 100 == 0: env.render()

            #concat_state(observation_,buffer_observation,num_agent,laser_mask)
            wrapper_state(new_observation_)
            concat_state(new_observation_, buffer_new_observation, num_agent,laser_mask)
            if 0:
                print('collid')

                render_img = env._env.Render(0)
                # aaa = cv2.resize(render*1,(500,500))
                lm = render_img.reshape(N * 50, N * 50, 3) * 1

                # if 1:
                # for i_a in range(self.num_agent):
                i_a = 0
                cv2.circle(lm, (int((new_observation_[i_a][4]-buffer_new_observation[i_a][0]) *100.0), int((new_observation_[i_a][5]-buffer_new_observation[i_a][1]) *100.0)), 10, (255, 0, 0), -1)
                draw_laser_mask = buffer_new_observation[:,120:].copy()
                x1,y1 = int(5*math.cos(buffer_new_observation[i_a][2])*100),int(5*math.sin(buffer_new_observation[i_a][2])*100)

                for i_a_i in range(num_agent):
                    if i_a!=i_a_i:
                        x1, y1 = int(1 * math.cos(buffer_new_observation[i_a_i][2])* 100) , int(1 * math.sin(buffer_new_observation[i_a_i][2])* 100)
                        #x1, y1 = int(1 * math.cos(buffer_new_observation[i_a_i][2])) * 100, int(1 * math.sin(buffer_new_observation[i_a_i][2])) * 100
                        cv2.line(lm, (int(buffer_new_observation[i_a_i][0] * 100.0), int(buffer_new_observation[i_a_i][1] * 100.0)), (int(buffer_new_observation[i_a_i][0] * 100.0) + x1,\
                                     int(buffer_new_observation[i_a_i][1] * 100.0) + y1), 0, 5)
                for ii in range(LASER_NUM):
                    for jj in range(DEGREE_NUM):
                        # cv2.circle(lm, (int(multi_ab[i_a][ii][jj][0] * 100), int(multi_ab[i_a][ii][jj][1] * 100)), 10,(0, 255, 0), -1)
                        if draw_laser_mask[i_a][DEGREE_NUM * ii + jj] == 1:
                            cv2.circle(lm, (int(mask_xy[i_a][ii][jj][0] * 100), int(mask_xy[i_a][ii][jj][1] * 100)),
                                       10, (0, 0, 0), -1)
                        else:
                            cv2.circle(lm, (int(mask_xy[i_a][ii][jj][0] * 100), int(mask_xy[i_a][ii][jj][1] * 100)),
                                       10, (0, 0, 255), -1)


                # lm[yy:(yy + 20), xx:(xx + 20), 1] = 0
                # lm[yy:(yy + 20), xx:(xx + 20), 2] = 1
                for i_a in range(num_agent):
                    cv2.putText(lm, str(i_a), (int(buffer_new_observation[i_a][0] *100.0) - 20, int(buffer_new_observation[i_a][1] *100.0) + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 6)
                    cv2.putText(lm, str(i_a), (int(buffer_new_observation[i_a][4] *100.0) - 20, int(buffer_new_observation[i_a][5] *100.0) + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 6)
                lm = cv2.resize(lm, (500, 500))
                cv2.imshow('aa', lm)
                cv2.waitKey(0)
                #cv2.imwrite('/hik/home/xinchao5/agv/RVO_Py_MAS/agv_obtacles/sim_xin_ok/libenv/data/aa.png', lm)
            #end_time = time.time()
            #during_time = end_time - start_time

                #wrapper_state(init_observation_1, init_observation, num_agent)
            #summary_str_all=[]
            summary_str_all = None
            for mmm  in range(num_agent):
                #all_actions[mmm].append(last_action[mmm]*1)
                all_states[mmm].append(new_observation_[mmm]*1)
                all_p[mmm].append(real_p[mmm]*1)
                #if mmm ==7:
                #_, summary_str = agent.run(observation_[mmm].copy(), last_action[mmm]*1, reward[mmm, 0]*1,\
                #                               terminal[mmm, 0]*1, real_terminal[mmm, 0]*1, new_observation_[mmm].copy(), mmm * 1)
                if 0:
                    _, summary_str = agent.run(buffer_observation[mmm].copy(), last_action[mmm] * 1, reward[mmm, 0] * 1, \
                                                           terminal[mmm, 0]*1, real_terminal[mmm, 0]*1, buffer_new_observation[mmm].copy(), mmm * 1)
                    if summary_str is not None:
                        summary_str_all = summary_str*1
                #_, summary_str = agent.run(observation_[mmm] * 1, last_action[mmm] * 1, reward[mmm, 0] * 1,
                #                           terminal[mmm, 0] * 1, real_terminal[mmm, 0] * 1, new_observation_[mmm],
                #                           mmm * 1)

            observation_= new_observation_.copy()
            #rvo_vflag_ = rvo_vflag*1
            mask_xy_ = mask_xy.copy()
            for uuu in range(num_agent):
                if terminal[uuu, 0] > 0 or real_terminal[uuu, 0] > 0:
                    observation_mm,endpoint_single,laser_mask,mask_xy_ = env.ireset(uuu)
                    if endpoint_single[0] not in endpoint:
                        endpoint.append(endpoint_single[0])
                    env.render()
                    wrapper_state(observation_mm)
                    observation_ = observation_mm.copy()
            # for ii in range(2):
            #    observation[ii,:] = new_observation[ii,:]*1
            # observation = new_observation
            if 0:
                if summary_str_all is not None:
                    summary_writer.add_summary(summary_str_all, agent.episode)





if __name__ == '__main__':
    main()
    print('done')
