"""A module used for post-processing of saved sessions.
"""

import sys, os, glob
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


def trim_to_shortest(signals):
    length = len(signals[0])
    for sig in signals:
        length = min(len(sig), length)
    
    for k in range(len(signals)):
        signals[k] = signals[k][:length]

    return signals

def get_signal_ma(signal, window_size=None):
    num = len(signal)
    window_size = window_size or np.max([int(num/15),5])
    signal_ma = moving_average(signal, window_size=window_size, mode='full')
    return signal_ma

## A class for plotting
# https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib/9890599
# https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html
# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/axes_demo.html#sphx-glr-gallery-subplots-axes-and-figures-axes-demo-py
# https://matplotlib.org/users/pyplot_tutorial.html
# https://docs.scipy.org/doc/scipy/reference/signal.html
# Averaging: https://becominghuman.ai/introduction-to-timeseries-analysis-using-python-numpy-only-3a7c980231af
class PostPlot:
    def __init__(self, loaders, output_dir = None):
        self.loaders = loaders
        self.varlogs = [l.getVarlogLoader() for l in self.loaders]
        self.output_dir = output_dir
        
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
        
        sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})
        # sns.set_context("paper")
        # sns.set(font_scale=1.5)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))  # create figure & 1 axis
        

        
        KEY = "/reward/"+mode+"/episodic"
        mean_loss_actors = [var[KEY]["sum"]/var[KEY]["num"] for var in self.varlogs]
        mean_loss_actors_ma = [get_signal_ma(signal, window_size=5) for signal in mean_loss_actors]

        mean_loss_actors_ma_trimmed = trim_to_shortest(mean_loss_actors_ma)
        
        
        episodes_all = [var[KEY]["episode"] for var in self.varlogs]
        episodes_all_trimmed = trim_to_shortest(episodes_all)
        episode = episodes_all_trimmed[0]
        
        epochs_all = [var[KEY]["epoch"] for var in self.varlogs]
        epochs_all_trimmed = trim_to_shortest(epochs_all)
        epoch = epochs_all_trimmed[0]

        abscissa = episode

        # frame = self.varlog[0][KEY]["frame"]
        # TODO: Choose the shortest signal from all loaders.

        # ax.plot(abscissa, mean_loss_actor, linewidth=1)
        stack = mean_loss_actors_ma_trimmed

        ax.plot(abscissa, np.mean(stack, axis=0), linewidth=1.5)
        ax.fill_between(abscissa, np.max(stack, axis=0), np.min(stack, axis=0), alpha=0.2) # color='gray'

        ax.set_title(mode.capitalize()+" time return")
        ax.set_ylabel("return (average of episodic rewards)")
        ax.set_xlabel("episodes")

        # Managing x axis limit
        ax.set(xlim=(np.min(abscissa), np.max(abscissa)))
        
        # Managing x axis ticks
        # Values
        ax.set(xticks=np.linspace(start=np.min(abscissa),stop=np.max(abscissa),num=4, endpoint=True))
        # Format
        ## xlabels = ['{:,.2f}'.format(x) + 'M' for x in ax.get_xticks()/1e6]
        ## ax.set_xticklabels(xlabels)

        # plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.tight_layout()

        if self.output_dir:
            path_to_output = self.output_dir
        else:
            path_to_output = self.loaders[0].getPlotsPath

        png_file = os.path.join(path_to_output, get_os_name(KEY)+".png")
        pdf_file = os.path.join(path_to_output, get_os_name(KEY)+".pdf")
        pkl_file = os.path.join(path_to_output, get_os_name(KEY)+".pkl")
        
        
        # fig.savefig(png_file, bbox_inches='tight', dpi=300)
        # We do not use bbox_inches to make all figure sizes consistent.
        fig.savefig(png_file, dpi=300)
        # fig.savefig(pdf_file, bbox_inches='tight')
        pickle.dump(ax, open(pkl_file,'wb'))
        plt.close(fig)
    


if __name__=="__main__":
    # import sys
    # print(" ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root-dir', metavar=('<path>'), default='/tmp/digideep_sessions', type=str, help="The root directory of sessions.")
    parser.add_argument('-i', '--session-names', metavar=('<path>'), type=str, nargs='+', action='append', required=True, help="Path to all input sessions in the root path. `--session-names session_*_*`")
    parser.add_argument('-o', '--output-dir', metavar=('<path>'), default='', type=str, help="Path to store the output plot.")
    args = parser.parse_args()

    args.session_names = [os.path.relpath(t, args.root_dir) for y in args.session_names for x in y for t in glob.glob(os.path.join(args.root_dir, x))]
    if args.output_dir == '' and len(args.session_names) == 1:
        args.output_dir = os.path.join(args.session_names[0], 'plots')
    output_dir = os.path.join(args.root_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Change the PYTHONPATH to load the saved modules for more compatibility.
    sys.path.insert(0, args.root_dir)

    loaders = []
    for s in args.session_names:
        loaders += [get_class(s + "." + "loader")]

    pp = PostPlot(loaders, output_dir)

    try:
        pp.plot_reward(mode="train")
    except Exception as ex:
        print(ex)
    
    try:
        pp.plot_reward(mode="test")
    except Exception as ex:
        print(ex)

