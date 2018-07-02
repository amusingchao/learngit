
import cv2
img = cv2.imread('../data/out.png', -1)
cv2.namedWindow('render',0)

import example
import numpy as np
import matplotlib.pyplot as plt
def run(show=True):
    a = plt.subplot(1, 1, 1)
    for i in range(4000):
        states = mm.Step()
        a1 = a.plot(i, states[0][3], 'ro--', label='line 1')
        render = mm.Render(1)
        if show:
            cv2.imshow('render', render.reshape(21*50,21*50,3))
            cv2.waitKey(1);
        print(states[:,-1], states[:,4])
        if max(states[:,-1]) > 0:
            plt.title("My matplotlib learning")
            plt.xlabel("X")
            plt.ylabel("Y")
            handles, labels = a.get_legend_handles_labels()
            a.legend(handles[::-1], labels[::-1])
            plt.savefig('plot_p.png')
            break;

    return states[:,-1]

mm = example.Env(img, 1)
print(mm.GetStartPort())
print(mm.GetEndPort())
print(mm.GetObsPort())
mm.Reset([0, 4], [3,2])

action01 = [500*18+250, 0, 0, 0]
action02 = [500*18+250, 0, 0, 0]
action03 = [500*18+250, 0, 0, 0]
action0 = [action01, action02, action03]

action11 = [500*8+250, 0, 0, 0]
action12 = [500*8+250, 0, 0, 0]
action13 = [500*8+250, 0, 0, 0]
action1 = [action11, action12, action13]

action = [action0, action1]

cmd_code = [1, 1]
cidx = [-1, -1]


for i, c in enumerate(cmd_code):
    if c > 0:
        cidx[i] += 1
        if cidx[i] >= 3:
            mm.iReset(i, i+0, 8)
            cidx[i] = 0
        mm.SetAction(i, np.array([action[i][cidx[i]]], dtype=np.float32))
cmd_code = run()

