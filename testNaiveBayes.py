import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.pyplot import Circle

from perceptron import Perceptron
from naiveBayes import NaiveBayes
import numpy as np
import math

import threading
import time


root = tkinter.Tk()
root.wm_title("Embedding in Tk")

fig = Figure(figsize=(6, 6), dpi=100)

t = np.arange(0, 3, .01)
plt = fig.add_subplot(1,1,1)


weights = np.zeros(3)

negData_x1 = []
negData_x2 = []
posData_x1 = []
posData_x2 = []
data = []


negPlot, = plt.plot(negData_x1, negData_x2, 'rs',  label='line 2',marker="o")
posPlot, = plt.plot(posData_x1, posData_x2, 'rs',  label='line 2',marker="P",color="green")

# plt.plot(t, 2 * np.sin(2 * np.pi * t))
plt.axis([-5,5,-5,5])
# fig.axes()1


canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

progressLabel = tkinter.Label(master=root, text="Progress ", justify = tkinter.CENTER)

naiveBayes = NaiveBayes()
    

typeOfPoint = tkinter.IntVar()
circle = None
selectPlot, = plt.plot([], [],  'o', mfc='none',markersize=15)
x = np.linspace(-5,5,100)
x2 = np.linspace(-5,5,100)



def on_key_press(event):
    print("you pressed {}".format(event.key))
    if event.key == "enter":
        if typeOfPoint.get() == 1:
            typeOfPoint.set(2)
        else:
            typeOfPoint.set(1)
    key_press_handler(event, canvas, toolbar)

def on_mouse_press(event):
    if typeOfPoint.get() == 1:
        data.append([event.xdata,event.ydata,1])
        posData_x1.append(event.xdata)
        posData_x2.append(event.ydata)
        posPlot.set_ydata(posData_x2)
        posPlot.set_xdata(posData_x1)
    else:
        data.append([event.xdata,event.ydata,-1])
        negData_x1.append(event.xdata)
        negData_x2.append(event.ydata)
        negPlot.set_ydata(negData_x2)
        negPlot.set_xdata(negData_x1)

    if len(posData_x1) > 0 and len(negData_x1) > 0:
        algoButton['state']= tkinter.NORMAL
    else:
        algoButton['state']= tkinter.DISABLED
        
    canvas.draw()
    canvas.flush_events()

    

    key_press_handler(event, canvas, toolbar)


canvas.mpl_connect("key_press_event", on_key_press)
canvas.mpl_connect("button_press_event", on_mouse_press)


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

def updateProgress(pointNumber,totalPoints):
    progressLabel['text'] = 'Progress %d of %d'%(pointNumber,totalPoints)

def startNaiveBayesThread():
    naiveBayes.setData(data)
    naiveBayes.handlers = [updateProgress]
    x0, x1 = np.meshgrid(
            np.linspace(-5, 5, int((5 - (-5)) * 50)),
            np.linspace(-5, 5, int((5 - (-5)) * 50)),
            )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = np.array(naiveBayes.predictClassVector(X_new))
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0,x1,zz,cmap=custom_cmap)

    canvas.draw()
    canvas.flush_events()

def _runNaiveBayes(): 
    threading.Thread(name='c1', target=startNaiveBayesThread, ).start()



button = tkinter.Button(master=root, text="Quit", command=_quit)
button.pack(side=tkinter.BOTTOM)

label = tkinter.Label(master=root,text="Select points first: ", justify = tkinter.LEFT,padx = 20)
label.pack(side=tkinter.TOP)


algoButton = tkinter.Button(master=root, text="Make Naive Bayes", command=_runNaiveBayes,state=tkinter.DISABLED)
algoButton.pack(side=tkinter.RIGHT)

rb = tkinter.Radiobutton(master=root,text="Positive",padx = 20,variable=typeOfPoint,value=1, justify = tkinter.LEFT,indicatoron=1)
rb.pack(side=tkinter.RIGHT)
rb.select()
tkinter.Radiobutton(master=root,text="Negative",padx = 20,variable=typeOfPoint,value=2, justify = tkinter.LEFT).pack(side=tkinter.RIGHT)
tkinter.Label(master=root, text="Type of point: ", justify = tkinter.LEFT).pack(side=tkinter.RIGHT)
progressLabel.pack(side=tkinter.TOP)
# Radiobutton(self.root,text="Autoplay",padx = 20,variable=self.rb,value=3, justify = LEFT).grid(row=4,column = 1)

tkinter.mainloop()

