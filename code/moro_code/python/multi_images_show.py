import argparse
import glob

import cv2
import numpy as np
import os


def show_in_one(images, show_size=(300, 300), blank_size=2, window_name="merge"):
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


if __name__ == '__main__':
    
    test_dir = "/home/xinchao/dd/explore/ORB_SLAM2/prj/python/save_fail2/local_map/"
    path = test_dir

    debug_images = []
    for infile in glob.glob(os.path.join(path, '*.*')):
        ext = os.path.splitext(infile)[1][1:]  # get the filename extenstion
        if ext == "png" or ext == "jpg" or ext == "bmp" or ext == "tiff" or ext == "pbm":
            print(infile)
            img = cv2.imread(infile)
            if img is None:
                continue
            else:
                debug_images.append(img)

    show_in_one(debug_images)
    cv2.waitKey(0)
    cv2.destroyWindow()
