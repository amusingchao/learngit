#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import sys
import argparse

from network import ActorCriticFFNetwork, ActorCriticLSTMNetwork
from training_thread import A3CTrainingThread
from scene_loader import THORDiscreteEnvironment as Environment

from utils.ops import sample_action

from constants import ACTION_SIZE
from constants import CHECKPOINT_DIR
from constants import NUM_EVAL_EPISODES
from constants import VERBOSE
from constants import USE_LSTM
from constants import EVAL_INIT_LOC
from constants import SAVE_FILE
from constants import TASK_TYPE
from constants import TEST_TASK_LIST
import cv2
import time
from utils.tools import SimpleImageViewer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
from decimal import getcontext, Decimal
x_draw=[]
z_draw=[]
r_draw=[]
x_z_list=[]
value_list=[]
NUM_nocollid=143
xz_numpy=np.zeros((NUM_nocollid*16+1,2)).astype('float32')
#global location
real_target_xz=[]
DEBUG_FORK=0
DRAW_VALUE_BEFROE=1

#current_id_list=[]
def plot2D(middle_index=None,scene_terminal=None,scene_init=None):
    #plt.xlabel('Z')
    # 设置Y轴标签
    #plt.ylabel('X')
    f = h5py.File('/home/xinchao/workplace/scene_data/living_room_08.h5', 'r')
    x_z=f['location'][()]
    #x_z.tolist()
    num_n=len(x_z)
    #ax.plot(x_draw, z_draw, color='green', linestyle='dashed', marker='o', markerfacecolor='red', markersize=3)
    fg=plt.figure(figsize=(20,20),dpi=98)

    ax = plt.subplot(111)
    ax.axis([-3, 12, -3, 12])    #[-3,3,-3,3]
    ax.set_ylabel("z", fontsize=14)
    ax.set_xlabel("x", fontsize=14)
    ax.plot(x_draw, z_draw, color='green', linestyle='dashed', marker='o', markerfacecolor='red', markersize=3)
    #ax.plot(x_draw,z_draw,color='green', linestyle='dashed', marker='o',markerfacecolor='red', markersize=3)#画连线图
    N = 50
    colors = np.random.rand(N)
     # 0 to 15 point radii

    #plt.scatter(x, y, c=colors, alpha=0.5)
    #ax.scatter(x_draw,z_draw)#画散点图
    getcontext().prec = 3
    ax.text(x_draw[0],z_draw[0], r'start')
    ax.text(x_draw[-1],z_draw[-1], r'end')
    ax.text(real_target_xz[0][0], real_target_xz[0][1] - 0.14, r'T')
    #ax.text(real_target_xz[0][0],real_target_xz[0][1]-0.14, r'T')
    if DRAW_VALUE_BEFROE==True:
        start_index=0
        end_index=middle_index
        #end_index = len(value_list)
    else:
        start_index = middle_index
        end_index=len(value_list)
    node_i=1

    for index_rr in range(start_index,end_index):
        #index_rr=index_rr+start_index
        ax.text(x_draw[index_rr] +0.11, z_draw[index_rr] + 0.11, str(node_i), fontsize=8)
        node_i=node_i+1
        if 1:
            if r_draw[index_rr]==0:
                ax.text(x_draw[index_rr]-0.25, z_draw[index_rr]+0.25,  Decimal(str(value_list[index_rr])) * 1,fontsize=8)
            if r_draw[index_rr]==90:
                ax.text(x_draw[index_rr]+0.02, z_draw[index_rr],  Decimal(str(value_list[index_rr])) * 1,fontsize=8)
            if r_draw[index_rr]==180:
                ax.text(x_draw[index_rr]-0.25, z_draw[index_rr]-0.25,  Decimal(str(value_list[index_rr])) * 1,fontsize=8)
            if r_draw[index_rr]==270:
                ax.text(x_draw[index_rr]-0.25, z_draw[index_rr],  Decimal(str(value_list[index_rr])) * 1,fontsize=8)
    bbox_props = dict(boxstyle="rarrow,pad=0.05", fc="cyan", ec="b", lw=2)
    if 1:
        for index in range(len(xz_numpy)-1):
            ax.add_patch(
                patches.Rectangle(
                    (xz_numpy[index+1][0]-0.25, xz_numpy[index+1][1]-0.25),  # (x,y)
                    0.5,  # width
                    0.5,
                    fill=False# height
                )
            )
        if 0:
            for i in range(num_n):
                ax.add_patch(
                    patches.Rectangle(
                        (x_z[i][0]-0.25, x_z[i][1]-0.25),  # (x,y)
                        0.5,  # width
                        0.5,
                        fill=False# height
                    )
                )
    save_file=SAVE_FILE+str(scene_terminal)+'_'+str(scene_init)+'.png'
    plt.savefig(save_file)
    plt.close('all')
    #plt.show()
def h5_to_numpy():

    f = h5py.File('/home/xinchao/workplace/scene_data/living_room_08.h5', 'r')
    location=f['location'][()]
    rotation=location=f['location'][()]
    for i in range(468):
        xz_numpy[i,:]=location[i,:]
if __name__ == '__main__':
  h5_to_numpy()
  h5_file = h5py.File('/home/xinchao/workplace/scene_data/living_room_08.h5', 'r')
  shortest_path_distances = h5_file['shortest_path_distance'][()]
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--time', help='Name of checkpoint file.',
                      default=None)
  args = parser.parse_args()

  device = "/cpu:0" # use CPU for display tool
  network_scope = TASK_TYPE
  list_of_tasks = TEST_TASK_LIST
  scene_scopes = list_of_tasks.keys()

  if USE_LSTM:
    global_network = ActorCriticLSTMNetwork(action_size=ACTION_SIZE,
                                          device=device,
                                          network_scope=network_scope,
                                          scene_scopes=scene_scopes)
  else:
    global_network = ActorCriticFFNetwork(action_size=ACTION_SIZE,
                                          device=device,
                                          network_scope=network_scope,
                                          scene_scopes=scene_scopes)

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

  if checkpoint and checkpoint.model_checkpoint_path:
    if args.time:
      checkpoint_path = "checkpoints/checkpoint-" + args.time
    else:
      checkpoint_path = checkpoint.model_checkpoint_path
    checkpoint_path='/home/xinchao'+checkpoint_path.split('/data/xinchao5')[1]
    saver.restore(sess, checkpoint_path)
    print("checkpoint loaded: {}".format(checkpoint_path))
  else:
    print("Could not find old checkpoint")

  scene_stats = dict()
  action_list = []
  for scene_scope in scene_scopes:

    scene_stats[scene_scope] = []
    for task_scope in list_of_tasks[scene_scope]:

      env = Environment({
        'scene_name': 'living_room_08',
        'terminal_state_id': int(task_scope),
        #'initial_state': EVAL_INIT_LOC,
      })
      real_target_xz.append([xz_numpy[int(task_scope)][0], xz_numpy[int(task_scope)][1]])
      ep_rewards = []
      ep_lengths = []
      ep_collisions = []

      scopes = [network_scope, scene_scope, task_scope]
      #time.sleep(5)
      a = np.arange(300 * 400 * 3)
      imx = a.reshape(300, 400, 3).astype('uint8')
      # im0 = img_from_thor()
      #env.last_event.frame.astype('float32')
      show_im=env.observation_target
      imx[:, :, 2] = show_im[:, :, 0]
      imx[:,:,1] = show_im[:,:,1]
      imx[:,:,0] = show_im[:,:,2]
      if 1:
          time.sleep(1)
          #cv2.imshow('target image', env.observation_target)
          cv2.imshow('target image', imx)
          cv2.waitKey(0)
      #viewer = SimpleImageViewer()
      #viewer.imshow(env.observation,None,str(0))
      #time.sleep(5)
      mean_sp=[]
      for i_episode in range(NUM_EVAL_EPISODES):

        env.reset()
        current_idindex=env.current_state_id
        sp_distance = shortest_path_distances[current_idindex][int(task_scope)]
        mean_sp.append(sp_distance)
        terminal = False
        ep_reward = 0
        ep_collision = 0
        ep_t = 0
        ep_action = []

        show_target = []
        max_value = []
        max_index = 0
        cur_id = []
        pi_values_list=[]
        while not terminal:

          pi_values, value_0= global_network.run_policy_and_value(sess, env.s_t, env.target, scopes)
          pi_values_list.append(pi_values)
          action = sample_action(pi_values)
          action_list.append(action)
          value_list.append(value_0)
          show_target.append(env.observation)
          cur_id.append(env.current_state_id)
          # current_id_list.append()
          x_draw.append(env.x)
          z_draw.append(env.z)
          r_draw.append(env.r)
          ep_action.append(action)
          env.step(action)
          env.update()
          time.sleep(0.3)
          max_value.append(value_0)
          #viewer.imshow(env.observation,None,str(value_0))
          cv2.imshow('observation image', env.observation)
          cv2.waitKey(1)
          terminal = env.terminal

          if ep_t == 500:
              break
          if env.collided: ep_collision += 1
          ep_reward += env.reward
          ep_t += 1
        time.sleep(2)
        if 1:
            pi_values, value_0 = global_network.run_policy_and_value(sess, env.s_t, env.target, scopes)
            pi_values_list.append(pi_values)
            action = sample_action(pi_values)
            action_list.append(action)
            value_list.append(value_0)
            show_target.append(env.observation)
            cur_id.append(env.current_state_id)
            x_draw.append(env.x)
            z_draw.append(env.z)
            r_draw.append(env.r)
            max_value.append(value_0)
            #print cur_id
        print '----------------'
        print task_scope,current_idindex
        print env.x_t,env.z_t,env.r_t
        print ep_collision
        print x_draw,len(x_draw)
        print z_draw,len(z_draw)
        print r_draw,len(r_draw)
        print pi_values_list,len(pi_values_list)
        print ep_action,len(ep_action)
        print max_value,len(max_value)
        print cur_id
        print sp_distance,len(action_list)-1
        #plot2D(len(r_draw),task_scope,current_idindex)
        x_draw[:] = []
        z_draw[:] = []
        r_draw[:] = []
        x_z_list[:] = []
        value_list[:] = []
        action_list[:]=[]
        pi_values_list[:]=[]
        if USE_LSTM:
          global_network.reset_state()
        ep_lengths.append(ep_t)
        ep_rewards.append(ep_reward)
        ep_collisions.append(ep_collision)
        if VERBOSE:
          print("episode #{} ends after {} steps".format(i_episode, ep_t))
          print(ep_action)
      print '----------------'
      print('evaluation: %s %s' % (scene_scope, task_scope))
      print('mean episode reward: %.2f' % np.mean(ep_rewards))
      print('mean episode length: %.2f' % np.mean(ep_lengths))
      print('mean sp length: %.2f' % np.mean(mean_sp))
      print('mean episode collision: %.2f' % np.mean(ep_collisions))
      mean_sp[:] = []
      f=open('./result_map/result.txt','a')
      f.write(scene_scope+' '+task_scope+' '+str(np.mean(ep_rewards))+' '+str(np.mean(ep_lengths))+' '+str(np.mean(ep_collisions))+'\n')
      #ep_lengths
      scene_stats[scene_scope].extend(ep_lengths)
f.close()
print('\nResults (average trajectory length):')
for scene_scope in scene_stats:
  print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats[scene_scope])))
