import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
  #ax = fig.add_subplot(1, 3, 1)
  #ax1 = fig.add_subplot(1, 3, 2)
  #ax2 = fig.add_subplot(1, 3, 3)

ax = fig.add_subplot(2, 2, 1)
ax.plot([1, 2, 3])
ax.legend(['A simple line'])
line, = ax.plot(0,0, label='Inline label')
# Overwrite the label by calling the method.
line.set_label('Label via method')
ax.legend()
plt.show()