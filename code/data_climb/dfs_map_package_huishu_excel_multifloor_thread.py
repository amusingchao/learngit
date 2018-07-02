#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stack import Stack

import robosims
import json
import cv2
import h5py
import random
import time
import numpy as np
import sys
MOVEAHEAD=0
ROTATELEFT=1
LOOKUP=2
MOVERIGHT=3
MOVELEFT=4
LOOKDOWN=5
ROTATERIGHT=6
MOVEBACK=7

Actions = ['MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'LookUp', 'LookDown', 'RotateRight', 'RotateLeft']
opposite_Action = ['MoveBack', 'MoveAhead', 'MoveRight',
                   'MoveLeft']  # , 'LookDown', 'LookUp', 'RotateLeft', 'RotateRight']
i_opposite = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}
rotation_bianli_0=['0.0','270.0','180.0','90.0']
rotation_bianli_90=['90.0','0.0','270.0','180.0']
rotation_bianli_180=['180.0','90.0','0.0','270.0']
rotation_bianli_270=['270.0','180.0','90.0','0.0']
#env = robosims.controller.ChallengeController(
    # Use unity_path=thor-201705011400-OSXIntel64.app/Contents/MacOS/thor-201705011400-OSXIntel64 for OSX
 #   unity_path='projects/thor-201705011400-Linux64',
 #   x_display="0.0"  # this parameter is ignored on OSX, but you must set this to the appropriate display on Linux
#)
#env.start(8090)
result = Stack()
goup_start_stack = Stack()
queue_noxuanzhuan_huishu = Stack()
map_obtacle=dict()
x_draw=[]
z_draw=[]
y_draw=[]
actioned = dict()
location = dict()
rotation = dict()
location_rotation = dict()
map_hdf5=dict()
pos_rot_list=[]
map=dict()
graph_lists = [[0 for i in range(8)] for j in range(10000)]
floortype_targetfile_targetposition=[]
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import xlwt
import os
def write_excel(input_list=None):
    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    DATA = (('学号', '姓名', '年龄', '性别', '成绩'),
            ('1001', 'A', '11', '男', '12'),
            ('1002', 'B', '12', '女', '22'),
            ('1003', 'C', '13', '女', '32'),
            ('1004', 'D', '14', '男', '52'),
            )
    for index_i,mm in enumerate(input_list):
        for index_j,uu in enumerate(mm.keys()):
            booksheet.write(index_i, index_j, str(mm[uu]))
    workbook.save('/data/xinchao5/multifllor/grade.xls')
def write_excel_field_pos_rotate(map_dict=None,excel_filename=None):
    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    DATA = (('学号', '姓名', '年龄', '性别', '成绩'),
            ('1001', 'A', '11', '男', '12'),
            ('1002', 'B', '12', '女', '22'),
            ('1003', 'C', '13', '女', '32'),
            ('1004', 'D', '14', '男', '52'),
            )
    for index_i, mm in enumerate(map_dict.keys()):
        booksheet.write(index_i, 0, str(mm))
        booksheet.write(index_i, 1, str(map_dict[mm]))
        booksheet.write(index_i, 2, float(map_dict[mm][0]))
        booksheet.write(index_i, 3, float(map_dict[mm][1]))
        booksheet.write(index_i, 4, float(map_dict[mm][2]))
    for index, mm in enumerate(map_obtacle.keys()):
        # print index,mm
        #obtacle_xzy = mm.split('[')[1].split(']')[0]
        #obtacle_x = float(obtacle_xzy.split(',')[0])
        #obtacle_z = float(obtacle_xzy.split(',')[1])
        booksheet.write(index_i+index+1, 1, str(mm))
        booksheet.write(index_i+index+1, 4, float(map_obtacle[mm]))

    filename_xls='/data/multifloor/multifloor_xls/'+str(scenename_file)
    if not os.path.exists(filename_xls):
        os.mkdir(filename_xls)
    #if not os.path.exists(filename_xls+'/'+excel_filename):
    workbook.save(filename_xls+'/'+excel_filename)
def hdf5_write(map_h5=None):
    num_map=len(map_h5)
    if os.path.exists('mytestfile.h5'):
        os.remove('mytestfile.h5')
    f = h5py.File("mytestfile.h5", "a")
    for index,key in enumerate(map_h5.keys()):
        dset2 = f.create_dataset(key, (1,), 'S15')
        dset2[0] = map_hdf5[key]
def plot2D(nn_inidex=None):
    #plt.xlabel('Z')
    # 设置Y轴标签
    #plt.ylabel('X')
    fg=plt.figure(figsize=(8,7),dpi=98)

    ax = plt.subplot(111)
    ax.axis([-4, 4, -4, 4])    #[-3,3,-3,3]
    ax.set_ylabel("z", fontsize=14)
    ax.set_xlabel("x", fontsize=14)
    ax.plot(x_draw,z_draw,color='green', linestyle='dashed', marker='o',markerfacecolor='red', markersize=3)#画连线图
    N = 50
    colors = np.random.rand(N)
     # 0 to 15 point radii

    #plt.scatter(x, y, c=colors, alpha=0.5)
    #ax.scatter(x_draw,z_draw)#画散点图
    ax.text(x_draw[0],z_draw[0], r'start')
    ax.text(x_draw[-1],z_draw[-1], r'end')
    bbox_props = dict(boxstyle="rarrow,pad=0.05", fc="cyan", ec="b", lw=2)
    if 1:
        for index,mm in enumerate(x_draw):
            ax.add_patch(
                patches.Rectangle(
                    (x_draw[index]-0.125, z_draw[index]-0.125),  # (x,y)
                    0.25,  # width
                    0.25,
                    fill=False# height
                )
            )
    #draw obtacle
        for index, mm in enumerate(map_obtacle.keys()):
            #print index,mm
            obtacle_xzy=mm.split('[')[1].split(']')[0]
            obtacle_x=float(obtacle_xzy.split(',')[0])
            obtacle_z = float(obtacle_xzy.split(',')[1])
            ax.add_patch(
                patches.Rectangle(
                    (obtacle_x - 0.125, obtacle_z - 0.125),  # (x,y)
                    0.25,  # width
                    0.25,
                    fill=True  # height
                )
            )
            #plt.show()
    #t = ax.text(z_draw[0], x_draw[0], "Direction", ha="center", va="center", rotation=0,size=3, bbox=bbox_props)
    #ax.annotate('111', xy=(0, 0.5), xytext=(0, 0.1), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    filename_maps = '/data/multifloor/multifloor_maps/' + str(scenename_file)
    if not os.path.exists(filename_maps):
        os.mkdir(filename_maps)
    #if not os.path.exists(filename_xls+'/'+excel_filename):
    #workbook.save(filename_xls+'/'+excel_filename)
    fg.savefig(filename_maps+'/'+onlytargetname_subfile+'_'+str(nn_inidex)+'.png', dpi=90, bbox_inches='tight')
    #plt.show()



def im_save(env_save=None, node_str=None):
    # env.initialize_target(target)
    a = np.arange(300 * 300 * 3)
    im = a.reshape(300, 300, 3).astype('float32')
    # im0 = img_from_thor()
    env_save.last_event.frame.astype('float32')
    im[:, :, 0] = env_save.last_event.frame[:, :, 2]
    im[:, :, 1] = env_save.last_event.frame[:, :, 1]
    im[:, :, 2] = env_save.last_event.frame[:, :, 0]
    #scenename_file, onlytargetname_subfile
    filename_img='/data/multifloor/multifloor_images/'+scenename_file+'/'+onlytargetname_subfile+'/'
    if not os.path.exists(filename_img):
        os.mkdir(filename_img)
    imgname =filename_img +node_str + '.png'
    if not os.path.exists(imgname):
        cv2.imwrite(imgname, im)

def int_camera_h(camera_h_input=None):

    if abs(camera_h_input-330)<1:
        camera_h_output=330
    if abs(camera_h_input-0)<1:
        camera_h_output=0
    if abs(camera_h_input-30)<1:
        camera_h_output=30
    if abs(camera_h_input-60)<1:
        camera_h_output=60
    return camera_h_output
def take_action(cur_node=None, actioned_dict=None):
    action_dict = dict()
    locate_rotate = []
    if actioned_dict[cur_node] == -1:
        action_index = 7  # 0
        action_take = Actions[action_index]
    else:
        action_index = actioned_dict[cur_node] + 1
        action_take = Actions[action_index]  # 外面的程序不能让8进来
    action_dict['action'] = action_take
    event = env.step(action=action_dict)

    actioned[cur_node] = actioned_dict[cur_node] + 1
    if not event.metadata['lastActionSuccess']:
        graph_lists[cur_node][action_index] = -1
        return -1
    else:
        locate_rotate = [env.last_event.metadata.get('agent').get('position'),
                         env.last_event.metadata.get('agent').get('rotation')]
        if locate_rotate in location_rotation.itervalues():
            # rotation[cur_node + 1] = env.last_event.metadata.get('agent').get('rotation')
            action_index = opposite_Action[action_index]
            action_take = Actions[action_index]  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            return cur_node
        next_node = cur_node + 1
        im_save(env, action_index, next_node)
        graph_lists[cur_node][action_index] = next_node
        return next_node
def save_to_map_hdf5(cur_node_with_h=None,next_node_with_h=None,forward_action=None):

    map_hdf5_key = cur_node_with_h + '_' + str(forward_action)
    if next_node_with_h=='-1':
        if map_hdf5_key not in map_hdf5.keys():
            map_hdf5[map_hdf5_key] = next_node_with_h
        return True
    if map_hdf5_key not in map_hdf5.keys():
        map_hdf5[map_hdf5_key]=next_node_with_h
    map_hdf5_key1 = next_node_with_h + '_' + str(7 - forward_action)
    if map_hdf5_key1 not in map_hdf5.keys():
        map_hdf5[map_hdf5_key1]=cur_node_with_h
    return True



def lookup_down_yuandian(root=None,camera_h_input=None):
    cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
    cur_camera_h=int_camera_h(cur_camera_h)
    start_camera_h=cur_camera_h*1
    action_dict=dict()
    cur_node_with_h=root+'_'+str(cur_camera_h)
    next_node_with_h = cur_node_with_h
    if start_camera_h==60:
        while (cur_camera_h != 330):
            cur_node_with_h = next_node_with_h
            action_take = 'LookUp'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            im_save(env,next_node_with_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKUP)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKUP) #collid
        while cur_camera_h !=60:
            cur_node_with_h=next_node_with_h

            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h =  env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h=root+'_'+str(cur_camera_h)
            im_save(env, next_node_with_h)
            save_done=save_to_map_hdf5(cur_node_with_h,next_node_with_h,LOOKDOWN)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKDOWN)  # collid
    if start_camera_h == 330:
        while (cur_camera_h != 60):
            cur_node_with_h = next_node_with_h
            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            im_save(env, next_node_with_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKDOWN)
        while cur_camera_h != 330:
            cur_node_with_h = next_node_with_h

            action_take = 'LookUp'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            im_save(env, next_node_with_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKUP)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKUP)
    if start_camera_h == 0:
        while (cur_camera_h != 60):
            cur_node_with_h = next_node_with_h
            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            im_save(env, next_node_with_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKDOWN)
        while cur_camera_h != 330:
            cur_node_with_h = next_node_with_h

            action_take = 'LookUp'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            im_save(env, next_node_with_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKUP)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKUP)
        while (cur_camera_h != 0):
            cur_node_with_h = next_node_with_h
            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            im_save(env, next_node_with_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
    if start_camera_h == 30:
        while (cur_camera_h != 60):
            cur_node_with_h = next_node_with_h
            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            im_save(env, next_node_with_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKDOWN)
        while cur_camera_h != 330:
            cur_node_with_h = next_node_with_h

            action_take = 'LookUp'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            im_save(env, next_node_with_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKUP)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKUP)
        while (cur_camera_h != 30):
            cur_node_with_h = next_node_with_h
            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            im_save(env, next_node_with_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
    return True,cur_camera_h

def fill_map_32to64(map_dict=None):
    map_keys=map_dict.keys()
    for key in map_keys:
        rotate_y=key.split('_')[1]
        node_index=key.split('_')[0]
        camera_h=key.split('_')[2]
        if rotate_y=='0.0':
            cur_node = key[0:-2]
            # str_key=str_key+'_'+str(ROTATERIGHT)
            next_node = node_index + '_' + '90.0' + '_'+camera_h
            save_to_map_hdf5(cur_node, next_node, ROTATERIGHT)
            next_node = node_index + '_' + '270.0' + '_'+camera_h
            save_to_map_hdf5(cur_node, next_node, ROTATELEFT)
        if rotate_y=='90.0':
            cur_node = key[0:-2]
            # str_key=str_key+'_'+str(ROTATERIGHT)
            next_node = node_index + '_' + '180.0' + '_'+camera_h
            save_to_map_hdf5(cur_node, next_node, ROTATERIGHT)
            next_node = node_index + '_' + '0.0' +'_'+ camera_h
            save_to_map_hdf5(cur_node, next_node, ROTATELEFT)
        if rotate_y=='180.0':
            cur_node = key[0:-2]
            next_node = node_index + '_' + '270.0' +'_' + camera_h
            save_to_map_hdf5(cur_node, next_node, ROTATERIGHT)
            next_node = node_index + '_' + '90.0' + '_'+camera_h
            save_to_map_hdf5(cur_node, next_node, ROTATELEFT)
        if rotate_y=='270.0':
            cur_node=key[0:-2]
            #str_key=str_key+'_'+str(ROTATERIGHT)
            next_node=node_index+'_'+'0.0'+'_'+camera_h
            save_to_map_hdf5(cur_node,next_node,ROTATERIGHT)
            next_node = node_index + '_' + '180.0' + '_'+camera_h
            save_to_map_hdf5(cur_node, next_node, ROTATELEFT)
def Rrotate_triple_up_down(root=None,root_h=None,camera_h_input=None):

    lookup_down_done,camera_h_output=lookup_down_yuandian(root,camera_h_input)
    action_dict=dict()
    action_take = 'RotateRight' # 外面的程序不能让8进来
    action_dict['action'] = action_take # 外面的程序不能让8进来
    cur_node_with_h =root+'_'+str(camera_h_output)
    next_node_with_h=root+'_'+str(camera_h_output)
    for i in range(3):
        cur_node_with_h=next_node_with_h
        event = env.step(action=action_dict)

        [position_x, position_z, rotation_y] = [env.last_event.metadata.get('agent').get('position')['x'],
                                                env.last_event.metadata.get('agent').get('position')['z'],
                                                env.last_event.metadata.get('agent').get('rotation')['y']]
        pos_rot_list = [position_x, position_z, rotation_y]

        real_root = root.split('_')[0] + '_' + str(rotation_y)
        next_node_with_h =real_root+'_'+str(camera_h_output)
        im_save(env,next_node_with_h)
        #save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, ROTATERIGHT)
        lookup_down_done, camera_h_output = lookup_down_yuandian(real_root, camera_h_input)
        result.push(real_root)
        goup_start_stack.push(real_root)

      #  for x in result.show_stack():
       #     print x
        if pos_rot_list in map.values():
            print list(map.keys())[list(map.values()).index(pos_rot_list)]
            print 'chongfu'
            print root
            #for x in result.show_stack():
            #    print x
           # print result.show_stack()
            plot2D()
        x_draw.append(position_x)
        z_draw.append(position_z)
        y_draw.append(rotation_y)
        map[real_root] = pos_rot_list
    fill_done=fill_map_32to64(map_hdf5)
    real_root_h=real_root+'_'+str(camera_h_output)
    return True,real_root,real_root_h,camera_h_output

def Rrotate_triple_up_down_gaile(root=None,root_h=None,camera_h_input=None):
    cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
    cur_camera_h = int_camera_h(cur_camera_h)
    start_camera_h = cur_camera_h * 1
    action_dict = dict()
    cur_node_with_h = root + '_' + str(cur_camera_h)
    next_node_with_h = cur_node_with_h
    if start_camera_h == 60:
        while (cur_camera_h != 330):
            action_dict = dict()
            action_take = 'RotateRight'  # 外面的程序不能让8进来
            action_dict['action'] = action_take  # 外面的程序不能让8进来
            cur_node_with_h = root + '_' + str(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            for i in range(3):
                cur_node_with_h = next_node_with_h
                event = env.step(action=action_dict)

                [position_x, position_z, rotation_y] = [env.last_event.metadata.get('agent').get('position')['x'],
                                                        env.last_event.metadata.get('agent').get('position')['z'],
                                                        env.last_event.metadata.get('agent').get('rotation')['y']]
                pos_rot_list = [position_x, position_z, rotation_y]

                real_root = root.split('_')[0] + '_' + str(rotation_y)
                next_node_with_h = real_root + '_' + str(cur_camera_h)
                save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, ROTATERIGHT)
                lookup_down_done, camera_h_output = lookup_down_yuandian(real_root, camera_h_input)
                if i==0:
                    result.push(real_root)
                    goup_start_stack.push(real_root)
            cur_node_with_h = next_node_with_h
            action_take = 'LookUp'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKUP)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKUP)  # collid
        while cur_camera_h != 60:
            cur_node_with_h = next_node_with_h

            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKDOWN)  # collid
    if start_camera_h == 330:
        while (cur_camera_h != 60):
            cur_node_with_h = next_node_with_h
            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKDOWN)
        while cur_camera_h != 330:
            cur_node_with_h = next_node_with_h

            action_take = 'LookUp'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKUP)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKUP)
    if start_camera_h == 0:
        while (cur_camera_h != 60):
            cur_node_with_h = next_node_with_h
            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKDOWN)
        while cur_camera_h != 330:
            cur_node_with_h = next_node_with_h

            action_take = 'LookUp'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKUP)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKUP)
        while (cur_camera_h != 0):
            cur_node_with_h = next_node_with_h
            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
    if start_camera_h == 30:
        while (cur_camera_h != 60):
            cur_node_with_h = next_node_with_h
            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKDOWN)
        while cur_camera_h != 330:
            cur_node_with_h = next_node_with_h

            action_take = 'LookUp'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKUP)
        cur_node_with_h = next_node_with_h
        save_done = save_to_map_hdf5(cur_node_with_h, '-1', LOOKUP)
        while (cur_camera_h != 30):
            cur_node_with_h = next_node_with_h
            action_take = 'LookDown'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
            cur_camera_h = env.last_event.metadata.get('agent').get('cameraHorizon')
            cur_camera_h = int_camera_h(cur_camera_h)
            next_node_with_h = root + '_' + str(cur_camera_h)
            save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, LOOKDOWN)
    return True, cur_camera_h

    lookup_down_done,camera_h_output=lookup_down_yuandian(root,camera_h_input)
    action_dict=dict()
    action_take = 'RotateRight' # 外面的程序不能让8进来
    action_dict['action'] = action_take # 外面的程序不能让8进来
    cur_node_with_h =root+'_'+str(camera_h_output)
    next_node_with_h=root+'_'+str(camera_h_output)
    for i in range(3):
        cur_node_with_h=next_node_with_h
        event = env.step(action=action_dict)

        [position_x, position_z, rotation_y] = [env.last_event.metadata.get('agent').get('position')['x'],
                                                env.last_event.metadata.get('agent').get('position')['z'],
                                                env.last_event.metadata.get('agent').get('rotation')['y']]
        pos_rot_list = [position_x, position_z, rotation_y]

        real_root = root.split('_')[0] + '_' + str(rotation_y)
        next_node_with_h =real_root+'_'+str(camera_h_output)
        save_done = save_to_map_hdf5(cur_node_with_h, next_node_with_h, ROTATERIGHT)
        lookup_down_done, camera_h_output = lookup_down_yuandian(real_root, camera_h_input)
        result.push(real_root)
        goup_start_stack.push(real_root)

      #  for x in result.show_stack():
       #     print x
        if pos_rot_list in map.values():
            print list(map.keys())[list(map.values()).index(pos_rot_list)]
            print 'chongfu'
            print root
            #for x in result.show_stack():
            #    print x
           # print result.show_stack()
            plot2D()
        x_draw.append(position_x)
        z_draw.append(position_z)
        y_draw.append(rotation_y)
        map[real_root] = pos_rot_list
    return True,real_root,camera_h_output

def Lrotate_once():
    action_dict=dict()
    action_take = 'RotateLeft' # 外面的程序不能让8进来
    action_dict['action'] = action_take
    event = env.step(action=action_dict)

def Rrotate_once():
    action_dict=dict()
    action_take = 'RotateRight' # 外面的程序不能让8进来
    action_dict['action'] = action_take
    event = env.step(action=action_dict)

def Moveback_nstep(root=None):
    back_done=False
    while not back_done:
        action_dict=dict()
        cur_y=root.split('_')[1]
        start_node=root.split('_')[0]
        huishu_path_start_index=queue_noxuanzhuan_huishu.show_stack().index(start_node)
        target_node=goup_start_stack.get_item().split('_')[0]   #str
        aa=queue_noxuanzhuan_huishu.get_i(huishu_path_start_index)
        while queue_noxuanzhuan_huishu.get_i(huishu_path_start_index) != target_node:
            #huishu_path_start_index=huishu_path_start_index-1
            cur_real_node = queue_noxuanzhuan_huishu.get_i(int(float(huishu_path_start_index)))
            cur_real_node_str = str(cur_real_node) + '_' + str(cur_y)
            cur_node_xyz = map[cur_real_node_str]

            next_real_node=queue_noxuanzhuan_huishu.get_i(int(float(huishu_path_start_index))-1)
            next_real_node_str=str(next_real_node)+'_'+str(cur_y)
            next_node_xyz = map[next_real_node_str]

            queue_noxuanzhuan_huishu.push(next_real_node_str.split('_')[0])

            cur_y = cur_node_xyz[2]
            cha_x = next_node_xyz[0] - cur_node_xyz[0]
            cha_z = next_node_xyz[1] - cur_node_xyz[1]
            x = abs(abs(cha_x) - 0.2499)
            z = abs(abs(cha_z) - 0.2499)
            if x<0.1:
                if cur_y==90.0:
                    if cha_x>0:
                        action_take = 'MoveAhead'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                    else:
                        action_take = 'MoveBack'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                if cur_y==270.0:
                    if cha_x>0:
                        action_take = 'MoveBack'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                    else:
                        action_take = 'MoveAhead'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                if cur_y == 180.0:
                    if cha_x>0:
                        action_take = 'MoveLeft'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                    else:
                        action_take = 'MoveRight'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                if cur_y == 0.0:
                    if cha_x>0:
                        action_take = 'MoveRight'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                    else:
                        action_take = 'MoveLeft'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)

            if z<0.1:
                if cur_y==90.0:
                    if cha_z>0:
                        action_take = 'MoveLeft'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                    else:
                        action_take = 'MoveRight'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                if cur_y==270.0:
                    if cha_z>0:
                        action_take = 'MoveRight'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                    else:
                        action_take = 'MoveLeft'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                if cur_y == 180.0:
                    if cha_z>0:
                        action_take = 'MoveBack'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                    else:
                        action_take = 'MoveAhead'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                if cur_y == 0.0:
                    if cha_z>0:
                        action_take = 'MoveAhead'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
                    else:
                        action_take = 'MoveBack'  # 外面的程序不能让8进来
                        action_dict['action'] = action_take
                        event = env.step(action=action_dict)
            huishu_path_start_index = huishu_path_start_index - 1
        cur_node=str(next_real_node)
        cur_root=cur_node + '_' + root.split('_')[1]
        pure_node = [x.split('_')[0] for x in goup_start_stack.show_stack()]
        if cur_node in pure_node:
            mm=float(cur_root.split('_')[1])
            nn=float(goup_start_stack.get_item().split('_')[1])
            cha_degree=mm-nn
            if cha_degree<0:
                n_times=int(abs(int(cha_degree)/90))
                for x in range(n_times):
                    action_dict = dict()
                    action_take = 'RotateRight'  # 外面的程序不能让8进来
                    action_dict['action'] = action_take
                    event = env.step(action=action_dict)
                    cur_root = cur_root.split('_')[0] + '_' + str(float(cur_root.split('_')[1])+90.0)
            elif cha_degree>0:
                n_times = int(abs(int(cha_degree) / 90))
                for x in range(n_times):
                    action_dict = dict()
                    action_take = 'RotateLeft'  # 外面的程序不能让8进来
                    action_dict['action'] = action_take
                    event = env.step(action=action_dict)
                    cur_root = cur_root.split('_')[0] + '_' + str(float(cur_root.split('_')[1])-90.0)
            else:
                back_done = True
            back_done=True

    x_draw.append(map[cur_root][0])
    z_draw.append(map[cur_root][1])
    y_draw.append(map[cur_root][2])
    return cur_root


def predict_up_xzy(current_xzy=None,action_up=None):
    current_x,current_z,current_y=current_xzy[0],current_xzy[1],current_xzy[2]
    if current_y==90:
        next_x=current_x+0.25
        next_z = current_z
    if current_y==180:
        next_x=current_x
        next_z = current_z-0.25
    if current_y==270:
        next_x=current_x-0.25
        next_z = current_z
    if current_y==0:
        next_x=current_x
        next_z = current_z+0.25
    next_y=current_y
    return [next_x,next_z,next_y]


def goup_or_not(root=None):
    flag=False
    #没有撞墙，有可能重复
    action_dict = dict()
    root_list =[]
    n_times=0
    if root.split('_')[1] =='0.0':
        root_list=rotation_bianli_0
    if root.split('_')[1] =='90.0':
        root_list=rotation_bianli_90
    if root.split('_')[1] =='180.0':
        root_list=rotation_bianli_180
    if root.split('_')[1] =='270.0':
        root_list=rotation_bianli_270
    for index,x in enumerate(root_list):
        cur_root=root.split('_')[0]+'_'+x
        if cur_root in goup_start_stack.show_stack():
            current_rotate_node = goup_start_stack.get_item()
            map_verify_xyz = map[cur_root]
            next_xyz = predict_up_xzy(map_verify_xyz, 'MoveAhead')
            map_all = []
            for x in map.values():
                cha_x=abs(x[0]-next_xyz[0])
                cha_z=abs(x[1] - next_xyz[1])
                if cha_x<0.1 and cha_z<0.1:
                    map_all.append([0,0])
                else:
                    map_all.append([x[0]-next_xyz[0],x[1]-next_xyz[1]])
            next_list=[]
            next_list=[0,0]
            if next_list in map_all:
                goup_start_stack.pop_value()
                Lrotate_once()
            else:
                action_take = 'MoveAhead'  # 外面的程序不能让8进来
                action_dict['action'] = action_take
                event = env.step(action=action_dict)
                action_success_flag = event.metadata['lastActionSuccess']
                if action_success_flag:
                    action_take = 'MoveBack'  # 外面的程序不能让8进来
                    action_dict['action'] = action_take
                    event = env.step(action=action_dict)
                    return True,cur_root
                else:
                    sss=[0,0,0]
                    mmm=goup_start_stack.get_item()
                    uuu=map[goup_start_stack.get_item()]
                    obtacle_y=uuu[2]
                    if obtacle_y==0.0:
                        sss[0]=uuu[0]
                        sss[1]=uuu[1]+0.25
                        sss[2]=obtacle_y
                    if obtacle_y==180.0:
                        sss[0] = uuu[0]
                        sss[1] = uuu[1] - 0.25
                        sss[2] = obtacle_y
                    if obtacle_y==90.0:
                        sss[0]=uuu[0]+0.25
                        sss[1]=uuu[1]
                        sss[2]=obtacle_y
                    if obtacle_y==270.0:
                        sss[0]=uuu[0]-0.25
                        sss[1] = uuu[1]
                        sss[2] = obtacle_y
                    map_obtacle[str(sss)]=-1
                    goup_start_stack.pop_value()
                    Lrotate_once()
                    continue
        else:
            Lrotate_once()
            continue
    #print'huishu'
    mm=env.last_event.metadata.get('agent').get('rotation')['y']
    nn= float(root.split('_')[1])
    cha_degree = mm - nn
    if cha_degree < 0:
        n_times = int(abs(int(cha_degree) / 90))
        for x in range(n_times):
            action_dict = dict()
            action_take = 'RotateRight'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
    elif cha_degree > 0:
        n_times = int(abs(int(cha_degree) / 90))
        for x in range(n_times):
            action_dict = dict()
            action_take = 'RotateLeft'  # 外面的程序不能让8进来
            action_dict['action'] = action_take
            event = env.step(action=action_dict)
    else:
        aa = True
    #Rrotate_once()
    return False,root

def goup_once(root=None):
    action_dict=dict()
    action_take = 'MoveAhead'  # 外面的程序不能让8进来
    action_dict['action'] = action_take
    event = env.step(action=action_dict)
    action_success_flag = event.metadata['lastActionSuccess']
    [position_x, position_z, rotation_y] = [env.last_event.metadata.get('agent').get('position')['x'],
                                            env.last_event.metadata.get('agent').get('position')['z'],
                                            env.last_event.metadata.get('agent').get('rotation')['y']]
    # x_draw.append(position_x)
    # z_draw.append(position_z)
    pos_rot_list = [position_x, position_z, rotation_y]
    # debug
    # if root=='15_0.0':
    #    print '15_0'
    goup_start_stack.pop_value()
    #real_root = str(int(root.split('_')[0]) + 1) + '_' + str(rotation_y)
    real_root = str(int(result.get_item().split('_')[0])+1) + '_' + str(rotation_y)
    result.push(real_root)
    goup_start_stack.push(real_root)
    queue_noxuanzhuan_huishu.push(real_root.split('_')[0])
    x_draw.append(position_x)
    z_draw.append(position_z)
    y_draw.append(rotation_y)
    if pos_rot_list in map.values():
        #for x in result.show_stack():
        #    print x
      #  print result.show_stack()
        print 'chongfu'
        print real_root
        plot2D()
    map[real_root] = pos_rot_list
    #map[real_root] = pos_rot_list
    return action_success_flag,real_root


def depth_first_search(root=None):
    print 'init before'
    print target
    env.initialize_target(target)
    print 'init target'
    [position_x,position_z,rotation_y,camera_h] = [env.last_event.metadata.get('agent').get('position')['x'],
                                          env.last_event.metadata.get('agent').get('position')['z'],
                                          env.last_event.metadata.get('agent').get('rotation')['y'],
                                          env.last_event.metadata.get('agent').get('cameraHorizon')]
    #rotation_y = env.last_event.metadata.get('agent').get('rotation')['y']
    pos_rot_list=[position_x,position_z,rotation_y]
    x_draw.append(position_x)
    z_draw.append(position_z)
    y_draw.append(rotation_y)
    real_root=str(root)+'_'+str(rotation_y)
    camera_h=int_camera_h(camera_h)
    real_root_h=str(root)+'_'+str(rotation_y)+'_'+str(camera_h)
    im_save(env,real_root_h)
    result.push(real_root)
    goup_start_stack.push(real_root)
    queue_noxuanzhuan_huishu.push(real_root.split('_')[0])
    map[real_root]=pos_rot_list
    # rotation[root] = env.last_event.metadata.get('agent').get('rotation')
    #mm=goup_start_stack.get_item()
    # mm=actioned.itervalues()
    num_step=1
    while goup_start_stack.show_stack():
        goup_done = False
        real_root = goup_start_stack.get_item()
        pos_rotate_x=map[real_root][0]
        pos_rotate_z=map[real_root][1]
        pos_1=[pos_rotate_x,pos_rotate_z,0.0]
        pos_2 = [pos_rotate_x, pos_rotate_z, 90.0]
        pos_3 = [pos_rotate_x, pos_rotate_z, 180.0]
        pos_4 = [pos_rotate_x, pos_rotate_z, 180.0]
        rotate_do = pos_1 not in map.values() or pos_2 not in map.values() or pos_3 not in map.values() or pos_4 not in map.values()
        # print rotate_do,real_root
        if rotate_do:
            rotate_done, real_root,real_root_h,camera_h= Rrotate_triple_up_down(real_root,real_root_h,camera_h)
            #hdf5_write(map_hdf5)


        if real_root.split('_')[0]=='5':    #137          #yinqinghuaile
            print '5'
            #delete_global_list_dict()
            #excel_filename = str(onlytargetname_subfile + '.xls')
            #write_excel_field_pos_rotate(map, excel_filename)
            #[(k, map[k]) for k in sorted(map.keys())]
        #    plot2D()
        goup_or_not_flag,real_root=goup_or_not(real_root)
        if not goup_start_stack.show_stack():
            return True
        if goup_or_not_flag:
            cur_node_h=real_root*1+'_'+str(camera_h)
            goup_flag,real_root=goup_once(real_root)
            next_node_h=real_root*1+'_'+str(camera_h)
            save_to_map_hdf5(cur_node_h,next_node_h,MOVEAHEAD)
            if goup_flag==False:
                print'go up wrong'
        else:
            a=1
            save_to_map_hdf5(cur_node_h, '-1', MOVEAHEAD)
            real_root=Moveback_nstep(real_root)
            #plot2D()#real_root=
        num_step=num_step+1
        #if num_step==250 or num_step==270:
        #    plot2D()
    return True



def delete_global_list_dict():
    result.clear_stack() #= Stack()
    goup_start_stack.clear_stack()# = Stack()
    queue_noxuanzhuan_huishu.clear_stack() # = Stack()
    map_obtacle.clear() #= dict()
    x_draw[:] = []
    z_draw[:] = []
    y_draw[:]= []
    actioned.clear()# = dict()
    location.clear() #= dict()
    rotation.clear() #= dict()
    location_rotation.clear() #= dict()
    map_hdf5.clear() #= dict()
    pos_rot_list[:] = []
    map.clear() #= dict()
    graph_lists[:]=[]
    #graph_lists = [[0 for i in range(8)] for j in range(10000)]
    #floortype_targetfile_targetposition[:] = []
def main():
    global t, target, env, fieldname
    global scenename_file, onlytargetname_subfile
    thread_index=sys.argv[1]
    print thread_index
    print 'thread name'
    env = robosims.controller.ChallengeController(
        # Use unity_path=thor-201705011400-OSXIntel64.app/Contents/MacOS/thor-201705011400-OSXIntel64 for OSX
        unity_path='projects/thor-201705011400-Linux64',#,
        x_display="0.0"  # this parameter is ignored on OSX, but you must set this to the appropriate display on Linux
    )
    port_index=8090+int(thread_index)
    env.start(port_index)
    bianliFloorName='FloorPlan'+thread_index
    bianli_scenename_list=[]
    print 'start start'
    with open("thor-challenge-targets/targets-train.json") as f:

        t = json.loads(f.read())
        for index_bianli,target in enumerate(t):
            bianli_scenename_list.append(target['sceneName'])
        start_index=bianli_scenename_list.index(bianliFloorName)
        end_index=start_index
        #write_excel(t)
        #for index,target in enumerate(t):
        while str(t[end_index]['sceneName'])==bianliFloorName:
            target=t[end_index]
            current_sceen=[target['sceneName'],target['targetImage'],target['targetPosition']]
	    print current_sceen
            if current_sceen in floortype_targetfile_targetposition:
                end_index = end_index + 1
                continue
            else:
                #print index
                scenename_file=target['sceneName']
                onlytargetname_subfile=target['targetObjectId']+'_'+str(end_index)
                floortype_targetfile_targetposition.append(current_sceen)
                #target = t[93]
                #fieldname = target['sceneName']
                search_done=depth_first_search(1)
                excel_filename = str(onlytargetname_subfile + '.xls')
                write_excel_field_pos_rotate(map, excel_filename)
                plot2D(end_index)
                delete_global_list_dict()
                end_index=end_index+1
    print'aa'


if __name__ == "__main__":
    main()
