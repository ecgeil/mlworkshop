"""
Interactively add points, fit a classifier, and plot the results.
"""

import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class InteractiveML:
    usage = """
        Click to place points. A model will be fit
        and drawn when there is at least one positive (black)
        and one negative (white) point.

        Keys:
            z: switch between positive/negative points
            x: clear last point placed
            c: clear all points
            m: next model
            n: previous model
            r: re-run model
            d: toggle probability plot (if available for model)
            h: display this help
            q: quit
    """

    def __init__(self, models):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(np.ones((1,1)), extent=(0,1,0,1), vmin=0, vmax=1,
            aspect='auto', cmap='gray')
        self.models = models
        self.cur_model = 0
        self.x = np.zeros((0,2))
        self.y = np.zeros(0)
        self.plot_prob = False
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkeydown)
        self.sign = 1
        self.redraw()
        self.ax.set_title("")
        self.ax.text(0.1, 0.1, self.usage)
    
    def onkeydown(self, event):
        if event.key == 'z':
            self.sign = -self.sign
        elif event.key == 'm':
            self.cur_model = (self.cur_model + 1) % len(models)
            self.redraw()
        elif event.key == 'n':
            self.cur_model = (self.cur_model - 1) % len(models)
            self.redraw()
        elif event.key == 'c':
            self.x = np.zeros((0,2))
            self.y = np.zeros(0)
            self.redraw()
        elif event.key == 'x' and len(self.y) > 0:
            self.y = self.y[:-1]
            self.x = self.x[:-1]
            self.redraw()
        elif event.key == 'r':
            self.redraw()
        elif event.key == 'd':
            self.plot_prob = not self.plot_prob
            self.redraw()
        elif event.key == 'h':
            self.ax.text(0.1, 0.1, self.usage)
            print self.usage
            self.fig.canvas.draw()
        elif event.key == 'q':
            sys.exit(0)
        
    def onclick(self, event):
        x, y = event.xdata, event.ydata
        if (0 <= x and x <= 1 and 0 <= y and y <= 1):
            if self.sign == 1:
                label = 1
            else:
                label = 0
            self.x = np.vstack((self.x, np.array([x,y])))
            self.y = np.append(self.y, label)
            self.redraw()
        
    def redraw(self):
        ax = self.ax
        ax.cla()
        name, model = self.models[self.cur_model]
        if np.unique(self.y).size > 1 and self.cur_model > 0:
            self.plot_fit(ax, model, self.x, self.y, prob=self.plot_prob)
        ax.scatter(np.append(self.x[:,0], -1),
            np.append(self.x[:,1], -1), 
            s=50, 
            c=np.append(self.y, 0),
            cmap='gray')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_title(name)
        self.fig.canvas.draw()

    def plot_fit(self, ax, model, xtrain, ytrain, prob=False, ngrid=100):
        model.fit(xtrain, ytrain)
        x = np.linspace(0, 1, ngrid)
        xv, yv = np.meshgrid(x, x)
        feature = np.column_stack((xv.flatten(), yv.flatten()))
        if hasattr(model , 'predict_proba') and prob:
            pred = model.predict_proba(feature)[:,1].reshape((ngrid, ngrid))
        else:
            pred = model.predict(feature).reshape((ngrid, ngrid))
        extent = (0, 1, 0, 1)
        ax.imshow(pred, extent=extent, vmin = 0, vmax = 1,origin='lower',
            aspect='auto', cmap='cool')

if __name__ == '__main__':
    #add your models here
    models = (("No model", None),
              ("Logistic Regression", LogisticRegression(C = 1e6, penalty = 'l2')),
              ("Logistic Regression (L1)", LogisticRegression(C = 1e6, penalty = 'l1')),
              ("Linear SVM", svm.SVC(kernel='linear', C=1e3)),
              ("Kernel SVM (polynomial)", svm.SVC(kernel='poly', C=1e6)),
              ("Kernel SVM (RBF)", svm.SVC(kernel='rbf', C=1e6)),
              ("Random Forest", RandomForestClassifier()))
    i = InteractiveML(models)
    plt.show()
    while (True):
        time.sleep(1)
