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
import networkx as nx
import matplotlib.image as mpimg
from constants import SCENE_NAME
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
    f = h5py.File('/home/xinchao/workplace/scene_data/'+SCENE_NAME+'.h5', 'r')
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
    ax.text(real_target_xz[0][0], real_target_xz[0][1] - 0.14, r'T',fontsize=2)
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
def plot_dongtai():
    #plt.close()  # clf() # 清图  cla() # 清坐标轴 close() # 关窗口
    plt.clf()

    IniObsX = 0000
    IniObsY = 4000
    IniObsAngle = 135
    #IniObsSpeed = 10 * math.sqrt(2)  # 米/秒
    print('开始仿真')
    for index in range(len(xz_numpy) - 1):
        ax.add_patch(
            patches.Rectangle(
                (xz_numpy[index + 1][0] - 0.25, xz_numpy[index + 1][1] - 0.25),  # (x,y)
                0.5,  # width
                0.5,
                fill=False  # height
            )
        )

    try:
        for t in range(len(show_target)):
            # 障碍物船只轨迹
            #obsX = IniObsX + IniObsSpeed * math.sin(IniObsAngle / 180 * math.pi) * t
            #obsY = IniObsY + IniObsSpeed * math.cos(IniObsAngle / 180 * math.pi) * t
            #ax.scatter(t, t, c='b', marker='.')  # 散点图
            ax.plot(x_draw[t], z_draw[t], color='green', linestyle='dashed', marker='x', markerfacecolor='red', markersize=3)
            #img = mpimg.imread('/home/xinchao/workplace/scene_images/'+str(t+3)+'.png')
            ax1.imshow(show_target[t])
            ax2.imshow(env.observation_target)
            #ax1.plot(t, t, color='green', linestyle='dashed', marker='o', markerfacecolor='red', markersize=3)
            # ax.lines.pop(1)  删除轨迹
            # 下面的图,两船的距离
            plt.pause(0.0001)
    except Exception as err:
        print(err)
def shortest_path_step(env, init_i=None,action_size=None):
    s_next_s_action = {}
    G = nx.DiGraph()
    paths=[]
    i=init_i
    for s in range(env.n_locations):
      for a in range(action_size):
        next_s = env.transition_graph[s, a]
        if next_s >= 0:
          s_next_s_action[(s, next_s)] = a
          G.add_edge(s, next_s)

    best_action = np.zeros((env.n_locations, action_size), dtype=np.float)
    for path in nx.all_shortest_paths(G, source=i, target=env.terminal_state_id):
        paths.append(path)
    return paths



def h5_to_numpy():

    f = h5py.File('/home/xinchao/workplace/scene_data/'+SCENE_NAME+'.h5', 'r')
    location=f['location'][()]
    rotation=location=f['location'][()]
    for i in range(468):        #468 livingroom08
        xz_numpy[i,:]=location[i,:]
if __name__ == '__main__':

  plt.close()
  plt.cla
  global fig,ax,ax1,ax2
  fig = plt.figure()
  #ax = fig.add_subplot(1, 3, 1)
  #ax1 = fig.add_subplot(1, 3, 2)
  #ax2 = fig.add_subplot(1, 3, 3)
  if 0:
      ax = fig.add_subplot(2, 2, 1)
      ax1 = fig.add_subplot(2, 2, 2)
      ax2 = fig.add_subplot(2, 2, 4)
      ax3 = fig.add_subplot(2, 2, 3)
  ax = fig.add_subplot(2, 2, 4)
  ax1 = fig.add_subplot(2, 2, 1)
  ax2 = fig.add_subplot(2, 2, 2)
  ax3 = fig.add_subplot(2, 2, 3)
  h5_to_numpy()
  h5_file = h5py.File('/home/xinchao/workplace/scene_data/'+SCENE_NAME+'.h5', 'r')
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
  global env
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
  flag_i=0
  for scene_scope in scene_scopes:

    scene_stats[scene_scope] = []
    for task_scope in list_of_tasks[scene_scope]:

      env = Environment({
        'scene_name': SCENE_NAME,
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
      if 0:
          time.sleep(1)
          #cv2.imshow('target image', env.observation_target)
          cv2.imshow('target image', imx)
          cv2.waitKey(0)
      #viewer = SimpleImageViewer()
      #viewer.imshow(env.observation,None,str(0))
      #time.sleep(5)
      mean_sp=[]
      for i_episode in range(NUM_EVAL_EPISODES):
        print'ok'
        env.reset(flag_i)
        if flag_i==0:
          flag_i=1
        elif flag_i==1:
          flag_i=0
        current_idindex=env.current_state_id
        sp_distance = shortest_path_distances[current_idindex][int(task_scope)]
        mean_sp.append(sp_distance)
        terminal = False
        ep_reward = 0
        ep_collision = 0
        ep_t = 0
        ep_action = []

        global show_target
        show_target = []
        max_value = []
        max_index = 0
        cur_id = []
        pi_values_list=[]
        while not terminal:
          x_draw.append(env.x)
          z_draw.append(env.z)
          r_draw.append(env.r)
          pi_values, value_0= global_network.run_policy_and_value(sess, env.s_t, env.target, scopes)
          pi_values_list.append(pi_values)
          action = sample_action(pi_values)
          ep_action.append(action)
          action_list.append(action)
          value_list.append(value_0)
          show_target.append(env.observation)
          cur_id.append(env.current_state_id)
          # current_id_list.append()


          env.step(action)
          env.update()
          #cv2.imshow('observation image', env.observation)
          #cv2.waitKey(1)
          #time.sleep(0.3)
          max_value.append(value_0)
          #viewer.imshow(env.observation,None,str(value_0))


          terminal = env.terminal

          if ep_t == 800:
              break
          if env.collided: ep_collision += 1
          ep_reward += env.reward
          ep_t += 1
        #time.sleep(2)
        #cv2.destroyAllWindows()
        init_ii=0
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
        print action_list
        print max_value,len(max_value)
        print cur_id
        print sp_distance,len(action_list)-1
        #plot_dongtai()
        paths=shortest_path_step(env,current_idindex,ACTION_SIZE)
        sp_x_0=[]
        sp_z_0 = []
        sp_x_1 = []
        sp_z_1 = []

        aaa = ax3.annotate('', xy=(0.6, 1), xytext=(0.6, 0.65),        #UP
                           arrowprops=dict(facecolor='white', shrink=0.3),
                           )
        aaa = ax3.annotate('', xy=(1, 0.6), xytext=(0.7, 0.6),
                           arrowprops=dict(facecolor='white', shrink=0.3),   #R
                               )

        aaa = ax3.annotate('', xy=(0.2, 0.6), xytext=(0.5, 0.6),
                           arrowprops=dict(facecolor='white', shrink=0.3),     #L
                               )

        aaa = ax3.annotate('', xy=(0.6, 0.3), xytext=(0.6, 0.65),
                           arrowprops=dict(facecolor='white', shrink=0.3),      #D
                               )
        bbb = ax3.annotate('', xy=(0.6, 1), xytext=(0.6, 0.65),  # UP
                           arrowprops=dict(facecolor='white', shrink=0.3),
                           )
        for nn in paths[0]:
            sp_x_0.append(xz_numpy[nn][0])
            sp_z_0.append(xz_numpy[nn][1])
        #for nn in paths[1]:
        #    sp_x_1.append(xz_numpy[nn][0])
        #    sp_x_1.append(xz_numpy[nn][1])
        if 1:

            #plt.clf()
            #
            ax.axis("equal")  # 设置图像显示的时候XY轴比例
            #ax.axis([-1, 7, -2, 11])
            #ax.axis([-1, 16, -1, 9]) #1028 # [-3,3,-3,3]
            # ax.set_ylabel("z", fontsize=14)
            # ax.set_xlabel("x", fontsize=14)
            ax.axis('off')
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            plt.grid(True)  # 添加网格
            plt.ion()
            ax.text(x_draw[0], z_draw[0], r'start',fontsize=2)
            ax.text(real_target_xz[0][0], real_target_xz[0][1] - 0.14, r'T',fontsize=2)
            #ax3.text(0, 0.4, r'Blue represents shortest path',fontsize=10,bbox=dict(facecolor='blue', alpha=0.5))
            #ax3.text(0, 0.2, r'Red represents our path',fontsize=10,bbox=dict(facecolor='red', alpha=0.5))
            ax.text(7, 8, r'shortest path', {'color': 'b', 'fontsize': 8})
            ax.text(7, 7, r'our path',{'color': 'r', 'fontsize': 8})
            #dots0= ax.plot(0, 0, color='blue', linestyle='dashed', marker='o',markerfacecolor='green',markersize=1)
            #dots0.set_label('Label via method')
            #ax.legend()
            ax1.text(0, 0, r'observation')
            ax2.text(0, 0, r'target')
            ax3.text(0.15, 0.9, r'Action',fontsize=10,bbox=dict(facecolor='green', alpha=0.5))
            print('开始仿真')
            if init_ii==0:
                time.sleep(5)
                init_ii=init_ii+1
            for t in range(len(sp_x_0)):
                lines = ax.plot(sp_x_0[t], sp_z_0[t], color='blue', linestyle='dashed', marker='o',
                                markerfacecolor='green',
                                markersize=1)
            for index in range(len(xz_numpy) - 1):
                ax.add_patch(
                    patches.Rectangle(
                        (xz_numpy[index + 1][0] - 0.25, xz_numpy[index + 1][1] - 0.25),  # (x,y)
                        0.5,  # width
                        0.5,
                        fill=False  # height
                    )
                )
            #plt.show()
            #uu=ax.text(15, 1, r'go up')

            try:
                for t in range(len(show_target)):
                    # 障碍物船只轨迹
                    # obsX = IniObsX + IniObsSpeed * math.sin(IniObsAngle / 180 * math.pi) * t
                    # obsY = IniObsY + IniObsSpeed * math.cos(IniObsAngle / 180 * math.pi) * t
                    # ax.scatter(t, t, c='b', marker='.')  # 散点图
                    dots=ax.scatter(x_draw[t], z_draw[t], c='r', marker='x')  # 散点图
                    bbb.remove()
                    #ax3.annotate("a", xy=(0.1, 1),arrowprops=dict(arrowstyle="->"))   #init xytext end xy
                    if 1:
                        if t==0:
                            bbb = ax3.annotate('', xy=(0.6, 1), xytext=(0.6, 0.65),  # UP
                                               arrowprops=dict(facecolor='white', shrink=0.3),
                                               )
                        else:
                            if action_list[t-1]==0:
                                bbb = ax3.annotate('', xy=(0.6, 1), xytext=(0.6, 0.65),  # UP
                                                   arrowprops=dict(facecolor='black', shrink=0.3),)
                            if action_list[t-1]==1:
                                bbb = ax3.annotate('', xy=(1, 0.6), xytext=(0.7, 0.6),
                                                   arrowprops=dict(facecolor='black', shrink=0.3),  # R
                                                   )
                            if action_list[t-1]==2:
                                bbb = ax3.annotate('', xy=(0.2, 0.6), xytext=(0.5, 0.6),
                                                   arrowprops=dict(facecolor='black', shrink=0.3),  # L
                                                   )

                            if action_list[t-1]==3:
                                bbb = ax3.annotate('', xy=(0.6, 0.3), xytext=(0.6, 0.65),
                                                   arrowprops=dict(facecolor='black', shrink=0.3),  # D
                                                   )
                    #ax.remove()

                    if 0:
                        if ep_action[t]==0:
                          uu.remove()
                          uu=ax.text(15, 1, r'go up')
                          #time.sleep(3)

                        if ep_action[t]==1:
                          #uu.remove()
                          uu.remove()
                          uu=ax.text(15, 1, r'rotate right')
                          #time.sleep(3)

                        if ep_action[t]==2:
                          #uu.remove()
                          uu.remove()
                          uu=ax.text(15, 1, r'rotate left')
                          #time.sleep(3)

                        if ep_action[t]==3:
                          #uu.remove()
                          uu.remove()
                          uu=ax.text(15, 1, r'go down')
                          #time.sleep(3)

                    #lines=ax.plot(x_draw[t], z_draw[t], color='green', linestyle='dashed', marker='o', markerfacecolor='red',
                    #        markersize=3)
                    # img = mpimg.imread('/home/xinchao/workplace/scene_images/'+str(t+3)+'.png')
                    ddd=ax1.imshow(show_target[t])
                    eee=ax2.imshow(env.observation_target)
                    # ax1.plot(t, t, color='green', linestyle='dashed', marker='o', markerfacecolor='red', markersize=3)
                    # ax.lines.pop(1)  删除轨迹
                    # 下面的图,两船的距离
                    plt.pause(0.02)
                    plt.savefig('result_images/'+str(t)+'.png',dpi=600)
                    #plt.pause(1)
            except Exception as err:
                print(err)
            ax.text(x_draw[-1], z_draw[-1], r'end')

            #for t in range(len(show_target)):
            #ax.lines.remove(lines[0])
            #ax.clf()
            #ax.p.pop(1)
            #删除轨迹
            #ax.remove()
        time.sleep(5)
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
