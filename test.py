import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import random
class RTplot():
    def __init__(self):
        self.x = []
        self.y = []
        self.c = []
        self.upCount = 0
        self.fig = plt.figure(figsize=(12,6))
        self.ax = plt.subplot()
        plt.autoscale(enable = True)

    def aniFunc(self, i):
        self.addPoint([random.randrange(1,100),random.randrange(1,100)])
        self.ax.cla()
        self.ax.scatter(self.x,self.y,c = self.c)
    
    def addPoint(self,list):
        if self.upcou