#from maze_env import Maze
#from RL_brain import DeepQNetwork
import os
import cv2
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from agv_no_obtacles_random import AGV
import example
import numpy as np
import tensorflow as tf
STATE_DIM = 7
ACTION_DIM = 60
"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
np.random.seed(1)
tf.set_random_seed(1)
V_NUM = 2
V_THETA_NUM = 3
STATE_CMD_CODE = 5
A0 = 0 #+
A1 = 1 #-
A2 = 2 #rotate left 90
A3 = 3 #rotate right -90
A4 = 4 #rotate right 180
A8 = 8
X_NUM = 0
Y_NUM  = 1
V_NUM = 2
V_THETA_NUM = 3
R_NUM = 4
Gx_NUM = 5
Gy_NUM = 6
# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            sess,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.network_params = tf.trainable_variables()


        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        #self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation,epsilon):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            if 0:
                choice_n = np.random.randint(0, 2)
                if choice_n == 0:
                    action = np.random.randint(0, self.n_actions-1)
                if choice_n == 1:
                    action = 8
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost,q_eval_now = self.sess.run([self._train_op, self.loss, self.q_eval],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return reward, q_eval_now, self.cost, q_target
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()




def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    episode_loss = tf.Variable(0.)
    tf.summary.scalar("Loss", episode_loss)

    episode_ave_max_q_target = tf.Variable(0.)
    tf.summary.scalar("TargetQmax Value", episode_ave_max_q_target)

    epison = tf.Variable(0.)
    tf.summary.scalar("epison", epison)

    summary_vars = [episode_reward, episode_ave_max_q, episode_loss, episode_ave_max_q_target,epison]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
def stop_status(obs_local,target_p):
    stop_flag = 0
    distance = abs(math.sqrt(math.pow((obs_local[X_NUM]-target_p[0]/1000.0), 2) + math.pow((obs_local[Y_NUM]-target_p[1]/1000.0), 2)))
    if distance>20:
        stop_flag = 1
    if (obs_local[Y_NUM]<0.1) or (obs_local[Y_NUM]>30) or(obs_local[X_NUM]<0.1) or (obs_local[X_NUM]>30):
        stop_flag = 1
    return stop_flag


def run_env(sess, mm,show, RL):
    step = 0

    #print(sess.run(RL.network_params[0]))
    summary_ops, summary_vars = build_summaries()
    writer = tf.summary.FileWriter('/hik/home/xinchao5/agv/RVO_Py_MAS/agv_prj/logs_no_obtacles_random/', sess.graph)
    sess.run(tf.global_variables_initializer()) #init after summary
    saver = tf.train.Saver(max_to_keep=10)
    epison = 0.1
    alpha = 1.005
    for episode in range(3000):
        # initial observation
        ep_reward = 0
        ep_ave_max_q = 0
        ep_ave_max_q_target = 0
        env = AGV(mm, show)
        observation,states = env.reset()
        print('start:',env.start_num,'end:',env.end_num)

        #while True:
        for j in range(1000):
            # fresh env
            #env.render()

            # RL choose action based on observation
            if states[0][STATE_CMD_CODE]<0:
                break
            if (step>100) and (step%10)==0:
                epison = epison*alpha
            if epison>1.0:
                epison = 1.0
            action = RL.choose_action(observation,epison)
            #last_action = action * 1
            # RL take action and get next observation and reward
            observation_, reward, done, states_ = env.step(observation,action,states)
            #print('action', action, observation_, reward, done)
            #('step',step,'take action', action, 'next ob', observation_)
            RL.store_transition(observation, action, reward, observation_)

            #if (step > 65) and (step % 5 == 0):   #200
            if step > 32:
                batch_reward, batch_q_eval,batch_loss,batch_q_target = RL.learn()
                ep_ave_max_q += np.amax(batch_q_eval)
                ep_ave_max_q_target += np.amax(batch_q_target)
                #print('aa')
            # swap observation
            observation = observation_
            states = states_
            ep_reward += reward
            flag_stop = stop_status(observation,env.target_p)
            if flag_stop:
                if step > 32:
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / float(j),
                        summary_vars[2]: batch_loss,
                        summary_vars[3]: ep_ave_max_q_target / float(j),
                        summary_vars[4]: epison,
                    })

                    writer.add_summary(summary_str, episode)
                    writer.flush()
                    print('too far: ',epison, 'step:', j, ep_reward, 'ep_ave_max_q: ', ep_ave_max_q / float(j), 'batch_loss', batch_loss, 'ep_ave_max_q_target', ep_ave_max_q_target / float(j))
                    saver.save(sess, '/hik/home/xinchao5/agv/RVO_Py_MAS/agv_prj/model_no_obtacles_random/model' + str(episode) + '.ckpt')
                break
            # break while loop when end of this episode
            if done:
                if step > 32:
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / float(j),
                        summary_vars[2]: batch_loss,
                        summary_vars[3]: ep_ave_max_q_target / float(j),
                        summary_vars[4]: epison,
                    })

                    writer.add_summary(summary_str, episode)
                    writer.flush()
                    print('terminal: ',epison, 'step:', j, ep_reward, 'ep_ave_max_q: ', ep_ave_max_q / float(j), 'batch_loss', batch_loss, 'ep_ave_max_q_target', ep_ave_max_q_target / float(j))
                    saver.save(sess, '/hik/home/xinchao5/agv/RVO_Py_MAS/agv_prj/model_no_obtacles_random/model' + str(episode) + '.ckpt')
                break
            if j==999:
                if step > 32:
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / float(j),
                        summary_vars[2]: batch_loss,
                        summary_vars[3]: ep_ave_max_q_target / float(j),
                        summary_vars[4]: epison,
                    })
                    writer.add_summary(summary_str, episode)
                    writer.flush()
                    print('999: ',epison, ep_reward, 'ep_ave_max_q: ', ep_ave_max_q / float(j), 'batch_loss', batch_loss,'ep_ave_max_q_target', ep_ave_max_q_target / float(j))
                    saver.save(sess, '/hik/home/xinchao5/agv/RVO_Py_MAS/agv_prj/model_no_obtacles_random/model' + str(episode) + '.ckpt')
                break
            step += 1

    # end of game
    print('game over')
    #env.destroy()


if __name__ == "__main__":
    # maze game
    with tf.Session() as sess:
        img = cv2.imread('data/out.png', -1)
        mm = example.Env(img, 2)
        show=False
        #env = AGV(mm,show)
        action_bound = np.array([2.0], np.float32)

        state_dim = STATE_DIM
        action_dim = ACTION_DIM
        RL = DeepQNetwork(sess,ACTION_DIM, STATE_DIM,
                          learning_rate=0.01,
                          reward_decay=0.1,   #0 dor random
                          e_greedy=0.5,
                          replace_target_iter=200,
                          memory_size=20000,
                          # output_graph=True
                          )
        run_env(sess,mm,show,RL)