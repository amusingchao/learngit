# -*- coding:utf-8 -*
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

reload(sys)

sys.setdefaultencoding("utf8")

left_image_file = '/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/left/'
seg_image_file = '/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/seg_data/'
right_image_file = '/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/right/'
local_map_image_file = '/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/local_map/'
global_map_image_file = '/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/path/'
depth_image_file = '/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/debug_data/'
path_image_file = '/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/path/'

final_images_file = '/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/final.png'
map_result_image_file = '/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/map_result/'
list_all = os.listdir('/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/local_map/')



list_pure = []
for list_sub in list_all:
  list_pure .append(list_sub.split('_')[0])
#print list_all
#print list_pure 

def find_local_map(input_file_name = None):
  #for i in range(307):
  real_index = list_pure.index(input_file_name)
  real_local_map = list_all[real_index]
    #print real_local_map
  return real_local_map
  
def show_in_one(images, show_size=(480,1280), blank_size=2, window_name="merge"):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("ingnore count %s" % (max_count - count))
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, merge_img)
  


if  __name__ == '__main__':
  
  debug_images = []


  for i in range(307):
    local_file_name = str(i+1) + '.png'
    real_file_name = find_local_map(local_file_name)
    print real_file_name
    left_image1 = cv2.imread(left_image_file + local_file_name)

    right_image1 = cv2.imread(right_image_file + local_file_name)

    depth_image1= cv2.imread(depth_image_file + local_file_name)


    local_map_image1 = cv2.imread(local_map_image_file + real_file_name)

    path_map_image = cv2.imread(path_image_file + local_file_name)

    map_result_image = cv2.imread(global_map_image_file+ local_file_name)
    global_map_image1 = cv2.imread(map_result_image_file + local_file_name)
    
    seg_image = cv2.imread(seg_image_file + local_file_name)
    if 0:

        final_image = cv2.imread(final_images_file)

        left_image11 = cv2.imread('/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/left/3.png')

        right_image11 = cv2.imread('/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/right/3.png')
        
        final_image[(0+200):(480+200),(0+50):(640+50)] = left_image11 * 1

        final_image[(0 + 200):(480 + 200), (0 + 50+640+100):(640 + 50+640+100)] = right_image11 * 1
        title_image = cv2.imread('/home/xinchao/dd/explore/ORB_SLAM2/prj/python/title.png')
        title_image = cv2.resize(title_image, (1600, 150)) * 1
        final_image[20:170,500:2100] = title_image * 1
    #cv2.imwrite('final.png',final_image)

    h = local_map_image1.shape[0]
    w = local_map_image1.shape[1]
    c = local_map_image1.shape[2]
    for ii in range(h):
        for jj in range(w):
            if (local_map_image1[ii,jj,0] == 128):
              global_map_image1[ii,jj,0] = 255
              global_map_image1[ii,jj,1] = 0
              global_map_image1[ii,jj,2] = 0
            if (local_map_image1[ii,jj,0] == 255):
              global_map_image1[ii,jj,0] = 0
              global_map_image1[ii,jj,1] = 255
              global_map_image1[ii,jj,2] = 0
    map_result_image = map_result_image[200:600, 200:600] * 1
    map_result_image = cv2.resize(map_result_image, (1000, 1000)) * 1

    local_map_image1 = local_map_image1[200:600, 200:600] * 1
    local_map_image1 = cv2.resize(local_map_image1, (1000, 1000)) * 1


    global_map_image1 = global_map_image1[200:600, 200:600] * 1
    global_map_image1 = cv2.resize(global_map_image1, (1000, 1000)) * 1

    #cv2.line(global_map_image1, (0, 0), (511, 511), 255, 5)
    #cv2.imwrite('aa.png',global_map_image1)

    left_image = cv2.resize(left_image1, (640, 480))

    right_image = cv2.resize(right_image1, (640, 480))
    depth_image = cv2.resize(depth_image1, (640, 480))

    aa = np.ones(2600 * 2600 * 3)*255
    show_global_image = aa.reshape(2600, 2600, 3).astype('uint8')

    #for left and right
    bb = np.ones(480 * 1400 * 3) * 255
    left_right_image = bb.reshape(480, 1400, 3).astype('uint8')
    left_right_image[0:480,0:640] = left_image * 1
    left_right_image[0:480, 760:1400] = right_image * 1

    #cv2.imwrite('yuanlai.png',left_right_image)

    #for depth image

    cc = np.ones(480 * 1400 * 3) * 255
    depth_image = cc.reshape(480, 1400, 3).astype('uint8')
    depth_image[0:480, 380:1020] = depth_image1 * 1
    #cv2.imwrite('depth.png', depth_image)

    # for local_map image
    local_map_image2 = cv2.resize(local_map_image1, (480, 480))
    dd = np.ones(480 * 1400 * 3) * 255
    local_image = dd.reshape(480, 1400, 3).astype('uint8')
    
    ##local_image[0:480, 460:940] = local_map_image2 * 1
    local_image[0:480, 400:1040] = seg_image * 1
    #cv2.imwrite('local_image.png', local_image)

    #for left pat

    dd = np.ones(2400 * 1400 * 3) * 255
    left_part_image = dd.reshape(2400, 1400, 3).astype('uint8')

    left_part_image[200:680,0:1400] = left_right_image

    left_part_image[(2400 - 480 - 200 -480-150 - 70 - 30 - 10):(2400 - 480 - 200-150 - 70 - 30 - 10), 0:1400] = depth_image

    cv2.putText(left_part_image, 'depth image', (520, 1650 - 70 - 30 - 15), cv2.FONT_HERSHEY_COMPLEX, 2,
                (255, 0, 0), thickness=4, lineType=8)



    jiantou_image = cv2.imread('jiantou.jpg')
    jiantou_image1 = cv2.resize(jiantou_image ,(200,200))
    #wenzi_image = cv2.imread('wenzi.png')
    left_part_image[(1650-50-30):(1850-50-30), 600:800] = jiantou_image1

    left_part_image[(2400 - 480 - 100-50):(2400 - 100-50), 0:1400] = local_image

    left_part_image[(850-70):(1050-70), 600:800] = jiantou_image1

    cv2.putText(left_part_image, 'left image', (200,750), cv2.FONT_HERSHEY_COMPLEX, 2,
                (255, 0, 0), thickness=4, lineType=8)

    cv2.putText(left_part_image, 'right image', (850, 750), cv2.FONT_HERSHEY_COMPLEX, 2,
                (255, 0, 0), thickness=4, lineType=8)

    cv2.putText(left_part_image, 'local map', (520, 2350), cv2.FONT_HERSHEY_COMPLEX, 2,
                (255, 0, 0), thickness=4, lineType=8)
    #cv2.imwrite('left_part_image.png', left_part_image)

    #for right part
    ee = np.ones(2400 * 1000 * 3) * 255
    right_part_image = ee.reshape(2400, 1000, 3).astype('uint8')

    right_part_image[200:(200+1000), 0:1000] = map_result_image#global_map_image1

    right_part_image[200+1000+200:2400, 0:1000] = global_map_image1
    #cv2.imwrite('right_part_image.png', right_part_image)
    #for all part
    ff = np.ones(2600 * 2600 * 3) * 255
    all_part_image = ff.reshape(2600, 2600, 3).astype('uint8')
    all_part_image[0:2400, 0+50:1400+50] = left_part_image
    all_part_image[0:2400,(1500+50):(1500+1000+50)] = right_part_image
    cv2.putText(all_part_image, 'path', ((1950+50-10-10), 1280), cv2.FONT_HERSHEY_COMPLEX, 3,   \
                (255, 0, 0), thickness=4, lineType=8)
    cv2.putText(all_part_image, 'map', ((1950+50-10-10), 2470), cv2.FONT_HERSHEY_COMPLEX, 3, \
                (255, 0, 0), thickness=4, lineType=8)
    #cv2.imwrite('all_part_image.png',all_part_image)
    title_image = cv2.imread('/home/xinchao/dd/explore/ORB_SLAM2/prj/python/title.png')
    title_image = cv2.resize(title_image, (1600, 150)) * 1
    all_part_image[20:170, 500:2100] = title_image * 1

    cv2.imwrite(str(i+1)+'.png',all_part_image)
    if 0:

        local_map_image = cv2.resize(local_map_image1, (640, 480))
        global_map_image = cv2.resize(global_map_image1, (640, 480))

        a = np.arange(480 * 640 * 3)
        show_left_image = a.reshape(480, 640, 3).astype('uint8')
        show_left_image[:,:,0] = left_image[:,:,2]
        show_left_image[:, :, 1] = left_image[:, :, 1]
        show_left_image[:, :, 2] = left_image[:, :, 0]
        if 1:
            b = np.arange(480 * 640 * 3)
            show_depth_image = b.reshape(480, 640, 3).astype('uint8')
            show_depth_image[:, :, 0] = depth_image[:, :, 2]
            show_depth_image[:, :, 1] = depth_image[:, :, 1]
            show_depth_image[:, :, 2] = depth_image[:, :, 0]

        c = np.arange(480 * 640 * 3)
        show_global_map_image = c.reshape(480, 640, 3).astype('uint8')
        show_global_map_image[:, :, 0] = global_map_image[:, :, 2]
        show_global_map_image[:, :, 1] = global_map_image[:, :, 1]
        show_global_map_image[:, :, 2] = global_map_image[:, :, 0]

        #cv2.hconcat()


        #cv2.imshow('aa',left_image)
        #cv2.waitKey(0)
        debug_images.append(left_image)
        debug_images.append(local_map_image)
        ##image = np.array(df) # dataframe to ndarray
        #cv2.imshow('aa',image)
        #cv2.waitKey(0)
        show_global_map_image1 = show_global_map_image[20:350, 150:400]
        show_global_map_image2 = cv2.resize(show_global_map_image1, (2500, 2500))

        show_left_image2= cv2.resize(show_left_image, (800, 800))
        local_map_image2 = cv2.resize(local_map_image, (800, 800))
        show_depth_image2 = cv2.resize(show_depth_image, (800, 800))





        if True:
            plt.close()
            plt.cla
            global fig, ax, ax1, ax2
            fig = plt.figure()
            if 0:
                ax = fig.add_subplot(2, 3, 6)
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 3, 4)
                ax3 = fig.add_subplot(2, 3, 5)
            if 1:
                ax = fig.add_subplot(2, 2, 4)
                ax1 = fig.add_subplot(2, 2, 1)
                ax2 = fig.add_subplot(2, 2, 2)
                ax3 = fig.add_subplot(2, 2, 3)
            if 0:
                ax = fig.add_subplot(4, 3, 12)
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(4, 3, 10)
                ax3 = fig.add_subplot(4, 3, 11)
            ax.axis('off')
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            ddd = ax2.imshow(show_left_image2)
            eee = ax.imshow(local_map_image2)
            fff = ax3.imshow(show_depth_image2)
            ggg = ax1.imshow(show_global_map_image2)

            ax1.text(500, 0, r'global map', {'color': 'b', 'fontsize': 8})
            ax2.text(250, 0, r'left image', {'color': 'b', 'fontsize': 8})
            ax3.text(250, 0, r'depth image', {'color': 'b', 'fontsize': 8})
            ax.text(250, 0, r'right image', {'color': 'b', 'fontsize': 8})
            #ax.text(7, 7, r'our path', {'color': 'r', 'fontsize': 8})
            # ax1.plot(t, t, color='green', linestyle='dashed', marker='o', markerfacecolor='red', markersize=3)
            # ax.lines.pop(1)  删除轨迹
            # 下面的图,两船的距离
            #plt.pause(0.02)
            #label = ["First", "Second", "Third"]
            #plt.legend(label, loc=0, ncol=2)
            ax1.add_patch(
                patches.Rectangle(
                    (0.1, 0.1),  # (x,y)
                    50,  # width
                    50,
                    color="#0000FF",
                    alpha=0.5,
                    # height
                )
            )
            ax1.text(55, 25, r'current obtacles', {'color': 'w', 'fontsize': 8})

            ax1.add_patch(
                patches.Rectangle(
                    (0.1, 55),  # (x,y)
                    50,  # width
                    50,
                    color="#00FF00",
                    alpha=0.5,
                    # height
                )
            )
            ax1.text(55, 105, r'current passible', {'color': 'w', 'fontsize': 8})
            if 0:
                ax1.add_patch(
                    patches.Rectangle(
                        (0.1, 110),  # (x,y)
                        50,  # width
                        50,
                        color="#00F000",
                        alpha=0.5,
                        # height
                    )
                )
                ax1.text(55, 105, r'obtacles', {'color': 'w', 'fontsize': 8})


            plt.savefig(str(i) + '.png', dpi=600)

  
#sfor i in range(307):
