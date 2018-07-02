# -*- coding:utf-8 -*
import numpy as np   
import matplotlib.pyplot as plt   
  
x = np.arange(1, 11, 1)  
  
# plt.plot(x, x * 2, label = "First")  
# plt.plot(x, x * 3, label = "Second")  
# plt.plot(x, x * 4, label = "Third")  
  
# # loc 设置显示的位置，0是自适应  
# # ncol 设置显示的列数  
# plt.legend(loc = 0, ncol = 2)  
  
# 也可以这样指定label  
label = ["First", "Second", "Third"]  
#plt.plot(x, x * 2)  
#plt.plot(x, x * 3)  
#plt.plot(x, x * 4)  
plt.legend(label, loc = 1, ncol = 2)  
plt.show()  
