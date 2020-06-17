from digideep.environment.common.vec_env import VecEnvWrapper
from digideep.environment.common.monitor import ResultsWriter
import numpy as np
import time

from digideep.utility.logging import logger
from ascii_graph import Pyasciigraph
from ascii_graph.colors import Red
from collections import OrderedDict


from bashplotlib.scatterplot import plot_scatter


class Histogram:
    def __init__(self, name, limit=None, bins=0, categories={}):
        if categories:
            if limit or (bins != 0):
                logger.fatal("Only 'categories' or '(limit,bins)' can be specified at a time.")
                exit()
            self.categoric = True
        else:
            if not (limit and (bins > 0)):
                logger.fatal("At least one of 'categories' or '(limit,bins)' should be specified.")
                exit()
            self.categoric = False

        
        self.name = name
        self.limit = limit
        self.bins = bins
        self.categories = categories

        if self.categoric:
            self.data = OrderedDict()
            for c in categories:
                self.data[c] = [[0],[0, Red]]
        else:
            self.data = OrderedDict()
            lbound = limit[0]
            ubound = limit[1]
            steps = (ubound - lbound) / bins
            self.keys = {}

            for i in range(bins):
                name = "({:4.2f},{:4.2f})".format(lbound + steps * i, lbound + steps * (i+1))
                self.keys[i] = name
                self.data[name] = [[0],[0, Red]]
    
    def add(self, value, success=False):
        if self.name == "Rewards":
            success = True
        if self.categoric:
            if not value in self.data:
                logger.fatal("For {}, categoric value ({}) not in existing categories.".format(self.name, value))
                return
            if success:
                self.data[value][0][0] += 1
            self.data[value][1][0] += 1
            
        else:
            lbound = self.limit[0]
            ubound = self.limit[1]
            bins = self.bins

            if (value < lbound) or (value > ubound):
                logger.fatal("For {}, value not in range: {:4.2f} ({:4.2f}, {:4.2f})".format(self.name, value, lbound, ubound))
                return
            i = int((value - lbound) / (ubound - lbound) * bins)

            if i == bins:
                i -= 1
            name = self.keys[i]

            if success:
                self.data[name][0][0] += 1
            self.data[name][1][0] += 1
            
    def plot(self):
        graph = Pyasciigraph()
        print()
        for line in graph.graph(self.name, list(self.data.items())):
            print(line)
        print()
        print("-"*80)



class Scatter:
    def __init__(self, name, xlim=None, ylim=None):
        self.name = name
        # self.xlim = xlim
        # self.ylim = ylim
        self.data_x = []
        self.data_y = []
    
    def add(self, x, y):
        self.data_x.append(x)
        self.data_y.append(y)

    def plot(self):
        plot_scatter(None, self.data_x, self.data_y, size=20, pch='x', colour="red", title=self.name)


    
