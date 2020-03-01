import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.pyplot import Circle

from logisticRegression import LogisticRegression
import numpy as np
import math


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

logisticRegression = LogisticRegression()
    

typeOfPoint = tkinter.IntVar()
circle = None
selectPlot, = plt.plot([], [],  'o', mfc='none',markersize=15)
x = np.linspace(-8,8,100)
x2 = np.linspace(-8,8,100)
weightLine, = plt.plot(x, logisticRegression.calculate_boundary(x), '-r', label='y=2x+1',color = 'black', linewidth=3)



def fireResult(weights, finished):
    if not finished:
        weightLine.set_xdata(x)
        weightLine.set_ydata(logisticRegression.calculate_boundary(x))
    else:
        x0, x1 = np.meshgrid(
            np.linspace(-5.0, 5.0, int((5.0 - (-5.0)) * 30)),
            np.linspace(-5.0, 5.0, int((5.0 - (-5.0)) * 30)),
            )
        X_new = np.c_[x0.ravel(), x1.ravel()]

        y_predict = np.array(logisticRegression.predict(X_new))
        
        zz = y_predict.reshape(x0.shape)

        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

        plt.contourf(x0,x1,zz,cmap=custom_cmap)

    # plt.plot([dataPoints[0]], [dataPoints[1]],  'o', mfc='none',markersize=15)
    canvas.draw()
    canvas.flush_events() 



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
def _runPerceptron():
    print(data) 
    logisticRegression.setData(data)
    logisticRegression.solve()
    # dataNP = 
logisticRegression.handlers = [fireResult]

button = tkinter.Button(master=root, text="Quit", command=_quit)
button.pack(side=tkinter.BOTTOM)

label = tkinter.Label(master=root,text="Select points first: ", justify = tkinter.LEFT,padx = 20)
label.pack(side=tkinter.TOP)


algoButton = tkinter.Button(master=root, text="Run Logistic Regression", command=_runPerceptron,state=tkinter.DISABLED)
algoButton.pack(side=tkinter.RIGHT)

rb = tkinter.Radiobutton(master=root,text="Positive",padx = 20,variable=typeOfPoint,value=1, justify = tkinter.LEFT,indicatoron=1)
rb.pack(side=tkinter.RIGHT)
rb.select()
tkinter.Radiobutton(master=root,text="Negative",padx = 20,variable=typeOfPoint,value=2, justify = tkinter.LEFT).pack(side=tkinter.RIGHT)
tkinter.Label(master=root, text="Type of point: ", justify = tkinter.LEFT).pack(side=tkinter.RIGHT)
# Radiobutton(self.root,text="Autoplay",padx = 20,variable=self.rb,value=3, justify = LEFT).grid(row=4,column = 1)

tkinter.mainloop()

