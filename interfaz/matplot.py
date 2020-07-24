import math
from matplotlib.widgets  import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class MvPoints():
    def __init__(self, xdata, ydata,fig,axis,color=0,area=40,alpha=1):
        self.x=xdata
        self.y=ydata        
        self.points=[]
        try:
            if color>=0 or color<10000:    
                self.colors=np.ones(len(xdata))*color
                self.oldcolor=self.colors
            else:
                self.colors=color*2
                self.oldcolor=color*2
        except:
            self.colors=color*2
            self.oldcolor=color*2
        
        self.fig=fig
        self.ax=axis
        self.evnt=None
        self.alpha=alpha
        self.area=area
        self.scat=self.ax[0].scatter(self.x,self.y,s=self.area, c=self.colors,alpha=self.alpha)
        self.labels=[]
        self.labels2=[]
        for i in range(len(xdata)):
            self.labels.append(self.ax[0].annotate(i, (self.x[i], self.y[i])))
            self.labels2.append(self.ax[1].annotate(i, (self.x[i], self.y[i])))
        self.ax[0].set_xlim((-1.3,1.3))
        self.ax[0].set_ylim((-1.3,1.3))    
        
        self.rs = RectangleSelector(self.ax[0], self.line_select_callback,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True)
        
    def line_select_callback(self,eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        newx=[]
        newy=[]
        self.points=[]
        xacc=0
        yacc=0
        self.rs.set_visible(False)  
        for i in range(len(self.x)):
            newx.append(self.x[i])
            newy.append(self.y[i])    
            if x1 <self.x[i] and x2>self.x[i] and y2>self.y[i] and y1<self.y[i]:
                self.colors[i]=1
                self.points.append(i)
                xacc+=self.x[i]
                yacc+=self.y[i]
        n=len(self.points)
        if n!=0:
            #CENTER THE MOUSE MOVING ACCORDING THE POINTS
            self.x0=xacc/n
            self.y0=yacc/n
            ## REMPLACE THE CHANGES POINTS
            self.x=newx
            self.y=newy
            self.scat.remove()
            self.scat=self.ax[0].scatter(self.x,self.y,s=self.area, c=self.colors,alpha=self.alpha)
            self.ax[0].set_xlim((-1.3,1.3))
            self.ax[0].set_ylim((-1.3,1.3))
            self.onclck=self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.evtrelease=self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
            self.evnt=self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
            self.rs=None
        else:   
            self.rs = RectangleSelector(self.ax[0], self.line_select_callback,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True)

    def on_drag(self,event):
        xpress, ypress = event.xdata, event.ydata
        try:
            for i in self.points:
                try:
                    self.labels[i].remove()
                    self.labels[i]=self.ax[0].annotate(i, (self.x[i], self.y[i]))
                except:
                    print('x')
                #self.labels[i]=self.ax[0].annotate(i, (self.x[i], self.y[i]))
                self.x[i]=self.x[i]-self.x0+xpress
                self.y[i]=self.y[i]-self.y0+ypress
                
            self.x0=xpress
            self.y0=ypress
            self.scat.remove()
            self.scat=self.ax[0].scatter(self.x,self.y,s=self.area, c=self.colors,alpha=self.alpha)
            self.ax[0].set_xlim((-1.3,1.3))
            self.ax[0].set_ylim((-1.3,1.3))
            self.fig.canvas.draw()
        except:
            print(xpress,ypress)
        
    def onclick(self,event):
        self.fig.canvas.mpl_disconnect(self.evnt)
        self.points=[]
        
    def onrelease(self,event):
        self.rs = RectangleSelector(self.ax[0], self.line_select_callback,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True)
        self.fig.canvas.mpl_disconnect(self.onclck)
        self.fig.canvas.mpl_disconnect(self.evtrelease)
        self.colors=self.oldcolor.copy()
    def change(self,x,y):
        try:
            self.scat2.remove()
        except:
            print("First")
        self.ax[0].set_title("Result of DR with Mixture Kernel")
        self.ax[1].set_title("Expected DR")
        self.scat2=self.ax[1].scatter(self.x,self.y,s=self.area, c=self.colors,alpha=self.alpha)
        self.ax[1].set_xlim((-1.3,1.3))
        self.ax[1].set_ylim((-1.3,1.3))
        
        self.scat.remove()
        self.scat=self.ax[0].scatter(x,y,s=self.area, c=self.colors,alpha=self.alpha)
        self.ax[0].set_xlim((-1.3,1.3))
        self.ax[0].set_ylim((-1.3,1.3))
        for i in range(len(self.x)):
            self.labels[i].remove()
            self.labels[i]=self.ax[0].annotate(i, (x[i], y[i]))
            self.labels2[i].remove()
            self.labels2[i]=self.ax[1].annotate(i, (self.x[i], self.y[i]))
        self.x=x
        self.y=y
        
"""
x = range(10)
y = range(10)
area = 50 
fig, ax = plt.subplots()
mv=MvPoints(x,y,fig,ax)
plt.show()
"""
