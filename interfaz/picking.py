import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_title('click on points')

line, = ax.plot(np.random.rand(100), 'o', picker=5)  # 5 points tolerance

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    #ax.set_position(('data', 0))
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', points)
    x=ax.get_position()
    x.x0=1
    print('data',x.x0)
    ax.set_position(x)
fig.canvas.mpl_connect('pick_event', onpick)
print(ax.scatter.x)
plt.show()