"""A module used for post-processing of saved sessions.
"""

import sys, os
import argparse

# import time
import numpy as np
import pickle


## For loading a class by name
# from digideep.utility.toolbox import get_module
from digideep.utility.toolbox import get_class


## Plotting modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# sns.set_style('whitegrid')

## Generic Functions
def moving_average(array, window_size=5, mode='full'):
    weights = np.ones(window_size) / window_size
    ma = np.convolve(array, weights, mode=mode)
    if mode == 'full':
        return ma[:-window_size+1]
    elif mode=='same':
        return ma
    elif mode=='valid':
        print("Size is reduced!")
        return ma

def get_os_name(path):
    path = path.lstrip("/")
    path = path.replace("/", "_")
    return path

## A class for plotting
# https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib/9890599
# https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html
# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/axes_demo.html#sphx-glr-gallery-subplots-axes-and-figures-axes-demo-py
# https://matplotlib.org/users/pyplot_tutorial.html
# https://docs.scipy.org/doc/scipy/reference/signal.html
# Averaging: https://becominghuman.ai/introduction-to-timeseries-analysis-using-python-numpy-only-3a7c980231af
class PostPlot:
    def __init__(self, loader):
        self.loader = loader
        self.varlog = loader.getVarlogLoader()
        
        # KEY = "/explore/reward/train"
        # print("Keys=", self.varlog[KEY].keys())
        # Keys: ['epoch', 'frame', 'roll', 'std', 'num', 'min', 'max', 'sum']

    def plot_reward(self, mode="train"):
        """
        This function plots the reward function and saves the `.png`, `.pdf`, and `.pkl` files.
        
        To later reload the `.pkl` file, one can use the following (to change format, labels, etc.):

        Example:

            import matplotlib.pyplot as plt
            %matplotlib notebook
            import pickle
            ax = pickle.load( open( "path_to_pkl_figure.pkl", "rb" ) )
            ax.set_title("Alternative title")
            plt.show()
        
        See:
            https://stackoverflow.com/a/12734723

        """
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

        KEY = "/explore/reward/"+mode
        mean_loss_actor = self.varlog[KEY]["sum"] / self.varlog[KEY]["num"]
        num = len(mean_loss_actor)
        window_size = np.max([int(num/15),5])
        mean_loss_actor_ma = moving_average(mean_loss_actor, window_size=window_size, mode='full')
        epoch = self.varlog[KEY]["epoch"]
        frame = self.varlog[KEY]["frame"] / 1e6

        ax.plot(frame, mean_loss_actor, linewidth=1)
        ax.plot(frame, mean_loss_actor_ma, linewidth=1.5)
        ax.set_title(mode.capitalize()+" time return")
        ax.set_ylabel("return (sum of rewards)")
        ax.set_xlabel("million frames")
        # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        png_file = os.path.join(self.loader.getPlotsPath, get_os_name(KEY)+".png")
        pdf_file = os.path.join(self.loader.getPlotsPath, get_os_name(KEY)+".pdf")
        pkl_file = os.path.join(self.loader.getPlotsPath, get_os_name(KEY)+".pkl")
        fig.savefig(png_file, bbox_inches='tight', dpi=300)
        fig.savefig(pdf_file, bbox_inches='tight')
        pickle.dump(ax, open(pkl_file,'wb'))
        plt.close(fig)
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--session-dirs', metavar=('<path>'), default='/tmp/digideep_sessions', type=str, help="The root directory of sessions.")
    parser.add_argument('--session-name', metavar=('<name>'), type=str, help="The name of the saved session.")

    # parser.add_argument('--arg', metavar=('<pattern>'), nargs='?', const='', type=str, help="")
    # parser.add_argument('--arg', metavar=('<n>'), default=X, type=int, help="")
    # parser.add_argument('--arg', action="store_true", help="")
    args = parser.parse_args()

    
    # Change the PYTHONPATH to load the saved modules for more compatibility.
    sys.path.insert(0, args.session_dirs)
    loader = get_class(args.session_name + "." + "loader")
    pp = PostPlot(loader)

    pp.plot_reward(mode="train")
    pp.plot_reward(mode="test")




