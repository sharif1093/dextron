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
from digideep.utility.json_encoder import JsonDecoder

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
    def __init__(self, loaders, output_dir = None, **options):
        self.loaders = loaders
        self.output_dir = output_dir
        self.options = options
        # Keys: ['epoch', 'frame', 'episode',   'std', 'num', 'min', 'max', 'sum']

    def plot(self, key):
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
        fig, ax = self.init_plot()
        abscissa_key = self.options.get("abscissa_key", "episode") # episode | epoch | frame
        limits = [None, None, None, None]

        for subloaders in self.loaders:
            varlogs = [l.getVarlogLoader() for l in subloaders]
            new_limit = self.plot_stack(fig, ax, varlogs, key, abscissa_key)
            limits = self.update_plot_limits(limits, new_limit)
        
        self.set_plot_options(fig, ax, abscissa_key, limits)
        self.close_plot(fig, ax, key)


    def update_plot_limits(self, limits, new_limit):
        for i in [0, 2]: # x_min and y_min
            if limits[i] is not None:
                limits[i] = min(limits[i], new_limit[i])
            else:
                limits[i] = new_limit[i]
        for i in [1, 3]: # x_max and y_max
            if limits[i] is not None:
                limits[i] = max(limits[i], new_limit[i])
            else:
                limits[i] = new_limit[i]
        return limits

    def plot_stack(self, fig, ax, varlogs, key, abscissa_key):
        ###########################################
        ### Get ordinate and abscissa from keys ###
        ###########################################
        # Ordinate
        window_size = self.options.get("window_size", 5)
        ordinate_stack_ave = [var[key]["sum"]/var[key]["num"] for var in varlogs]
        ordinate_stack_min = [var[key]["min"] for var in varlogs]
        ordinate_stack_max = [var[key]["max"] for var in varlogs]

        ordinate_stack_ave_ma = [get_signal_ma(signal, window_size=window_size) for signal in ordinate_stack_ave]
        # Choose the shortest signal from all loaders. Because they are all going to be shown on the same plot.
        stack = trim_to_shortest(ordinate_stack_ave_ma)
        ordinate_max = np.max(stack,  axis=0)
        ordinate_min = np.min(stack,  axis=0)
        ordinate_ave = np.mean(stack, axis=0)
        
        # Abscissa
        abscissa_all = [var[key][abscissa_key] for var in varlogs]
        abscissa_all_trimmed = trim_to_shortest(abscissa_all)
        abscissa = abscissa_all_trimmed[0]

        ##################################
        ### Plot and fill the variance ###
        ##################################
        ax.plot(abscissa, ordinate_ave) # linewidth=1.5
        ax.fill_between(abscissa, ordinate_max, ordinate_min, alpha=0.2) # color='gray'

        xlim_min = np.min(abscissa)
        xlim_max = np.max(abscissa)
        ylim_min = np.min(ordinate_min)
        ylim_max = np.max(ordinate_max)
        return (xlim_min, xlim_max, ylim_min, ylim_max)


    def init_plot(self):
        ############################
        ### Overall plot options ###
        ############################
        context = self.options.get("context", "notebook") # notebook | paper
        font_scale = self.options.get("font_scale", 2)
        line_width = self.options.get("line_width", 2.5)
        width  = self.options.get("width", 10)
        height = self.options.get("height", 8)

        sns.set_context(context, font_scale=font_scale, rc={"lines.linewidth": line_width})
        # sns.set_context("paper")
        # sns.set(font_scale=1.5)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))  # create figure & 1 axis
        return fig, ax

    def set_plot_options(self, fig, ax, abscissa_key, limits):
        #########################
        ### Set plot settings ###
        #########################
        # Set labels and titles
        title_default  = ""
        title = self.options.get("title",  title_default)

        xlabel_default = abscissa_key
        xlabel = self.options.get("xlabel", xlabel_default)

        ylabel_default = "return (average of episodic rewards)"
        ylabel = self.options.get("ylabel", ylabel_default)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Managing axis limit
        xlim_min = self.options.get("xlim_min", limits[0])
        xlim_max = self.options.get("xlim_max", limits[1])
        ylim_min = self.options.get("ylim_min", limits[2])
        ylim_max = self.options.get("ylim_max", limits[3])

        ax.set(xlim=(xlim_min, xlim_max))
        ax.set(ylim=(ylim_min, ylim_max))

        # Managing x axis ticks
        xticks_num = self.options.get("xticks_num", 4)
        ax.set(xticks=np.linspace(start=xlim_min, stop=xlim_max, num=xticks_num, endpoint=True).astype(np.int))

        if abscissa_key == "episode" or abscissa_key == "epoch":
            xlabels = ['{:d}'.format(int(x)) for x in ax.get_xticks()]
            ax.set_xticklabels(xlabels)
        elif abscissa_key == "frame":
            # TODO: Consider other cases where abscissa can have "K" unit, etc.
            # frame_unit = self.options.get("frame_unit", "m")
            xlabels = ['{:,.2f}'.format(x) + 'M' for x in ax.get_xticks()/1e6]
            ax.set_xticklabels(xlabels)

        # Set the legends
        legends = self.options.get("legends", None)
        legends_option = self.options.get("legends_option", {})
        # loc="upper left"/"best"/"lower right" | ncol=2 | frameon=false
        if legends is not None:
            ax.legend(labels=legends, **legends_option)

    
    def close_plot(self, fig, ax, key):
        #####################################
        ### Fix layouts and save to files ###
        #####################################
        # plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.tight_layout()

        if self.output_dir:
            path_to_output = self.output_dir
        else:
            path_to_output = self.loaders[0][0].getPlotsPath

        png_file = os.path.join(path_to_output, get_os_name(key)+".png")
        pdf_file = os.path.join(path_to_output, get_os_name(key)+".pdf")
        pkl_file = os.path.join(path_to_output, get_os_name(key)+".pkl")
        
        fig.savefig(png_file, dpi=self.options.get("dpi", 300))
        # We do not use bbox_inches to make all figure sizes consistent.
        # fig.savefig(pdf_file, bbox_inches='tight')
        # fig.savefig(png_file, bbox_inches='tight', dpi=300)

        pickle.dump(ax, open(pkl_file,'wb'))
        plt.close(fig)

if __name__=="__main__":
    # import sys
    # print(" ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root-dir', metavar=('<path>'), default='/tmp/digideep_sessions', type=str, help="The root directory of sessions.")
    parser.add_argument('-i', '--session-names', metavar=('<path>'), type=str, nargs='+', action='append', required=True, help="Path to all input sessions in the root path. `--session-names session_*_*`")
    parser.add_argument('-o', '--output-dir', metavar=('<path>'), default='', type=str, help="Path to store the output plot.")
    
    parser.add_argument('--options', metavar=('<json dictionary>'), default=r'{}', type=JsonDecoder, help="Set the options as a json dict.")
    args = parser.parse_args()

    # Expand session_names
    for index in range(len(args.session_names)):
        args.session_names[index] = [os.path.relpath(t, args.root_dir) for y in args.session_names[index] for t in sorted(glob.glob(os.path.join(args.root_dir, y)))]

    # print("Commandline was:\n  ", " ".join(sys.argv[:]) )
    print("Added sessions are:\n  ", args.session_names)

    if args.session_names == [[]]:
        print("No sessions were added ...")
        sys.exit(1)
    
    if args.output_dir == '' and len(args.session_names) == 1:
        args.output_dir = os.path.join(args.session_names[0][0], 'plots')
    output_dir = os.path.join(args.root_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Change the PYTHONPATH to load the saved modules for more compatibility.
    # TODO: Why?
    sys.path.insert(0, args.root_dir)

    loaders = []
    for sublist in args.session_names:
        subloaders = []
        for s in sublist:
            subloaders += [get_class(s + "." + "loader")]
        loaders += [subloaders]

    pp = PostPlot(loaders, output_dir, **args.options)
    for key in args.options.get("key", ["/reward/train/episodic"]):
        pp.plot(key)
