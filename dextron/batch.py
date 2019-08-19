import shlex
import subprocess
import multiprocessing
from multiprocessing.pool import ThreadPool

import os, re, glob, shutil, argparse

from digideep.utility.stats import StatLogger
from digideep.pipeline.session import generateTimestamp

from tqdm import tqdm

def call_proc(cmd):
    """ This runs in a separate thread. """
    # print("cmd>", cmd)
    p = subprocess.Popen(shlex.split(cmd, posix=True), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)

class JobPool:
    def __init__(self, command_list, nproc=None):
        self.nproc = nproc or max(multiprocessing.cpu_count()-2, 1)
        self.pool = ThreadPool(self.nproc)
        self.command_list = command_list
        self.results = []

    def run(self):
        pbar = tqdm(total=len(self.command_list))
        def update(*a):
            pbar.update()
        # tqdm.write(str(a))
        for i in range(pbar.total):
            command = self.command_list[i]
            self.pool.apply_async(call_proc, args=(command,), callback=update)

        # for command in self.command_list:
        #     self.results.append(self.pool.apply_async(call_proc, (command,)))
        # Close the pool and wait for each running task to complete
        self.pool.close()
        self.pool.join()
        return self

    def print_out(self):
        for result, command in zip(self.results, self.command_list):
            out, _ = result.get()
            print("cmd>", command)
            print("."*85)
            print(out.decode("utf-8"))
            print("\n"+"="*85+"\n")
        return self

    def print_err(self):
        for result, command in zip(self.results, self.command_list):
            _, err = result.get()
            print("cmd>", command)
            print("."*85)
            print(err.decode("utf-8"))
            print("\n"+"="*85+"\n")
        return self

    def print_all(self):
        for result, command in zip(self.results, self.command_list):
            out, err = result.get()
            print("cmd>", command)
            print("."*85)
            print(out.decode("utf-8"))
            print("."*85)
            print(err.decode("utf-8"))
            print("\n"+"="*85+"\n")
        return self

def clean_spaces(command):
    return re.sub("\s\s+" , " ", command)

def lsdir(root):
    return [x for x in os.listdir(session_path) if os.path.isdir(os.path.join(session_path,x))]

def rmdir(path):
    os.system('rm -rf {}'.format(path))

def check_done(root, name):
    return os.path.exists(os.path.join(root, name, 'done.lock'))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session-path', metavar=('<path>'), type=str, default="/tmp/digideep_sessions", help="Path to session storage.")
    parser.add_argument('--reports-path', metavar=('<path>'), type=str, default="/reports", help="Path to reports storage.")
    parser.add_argument('--logs-path', metavar=('<path>'), type=str, default="/logs", help="Path to log files.")
    parser.add_argument('--instance', metavar=('<str>'), type=str, default="default", help="Name of running instance.")
    # parser.add_argument('--session-path', metavar=('<path>'), type=str, default="/scratch/sharif.mo/digideep_slurm/sessions/other", help="Path to session storage.")
    # parser.add_argument('--reports-path', metavar=('<path>'), type=str, default="/scratch/sharif.mo/digideep_slurm/reports", help="Path to reports storage.")
    parser.add_argument('--resume', action='store_true', help="Whether to resume already processed data or to remove the existing sessions_path and reports.")
    args = parser.parse_args()

    session_path = args.session_path
    reports_path = args.reports_path
    logs_path = args.logs_path
    instance = args.instance

    # print(os.environ)

    if args.resume and os.path.exists(session_path):
        # Get all directories in the session_path
        # Check if "done.lock" is inside.
        # If "YES", don't do that job again.
        # If "NO", remove that job and ket that to be done again.

        existing = set(lsdir(session_path))
        donelist = {x for x in existing if check_done(session_path, x)}
        incomplete = existing - donelist
        # Remove incomplete items:
        for x in incomplete:
            rmdir(os.path.join(session_path, x))
        print("Complete sessions:", donelist)
    else:
        donelist = {}
        # Complain if the directories already exist.
        # Ask user to remove the directories then continue.
        # This helps increase safety.
        if os.path.exists(session_path):
            raise Exception("The 'session_path' already exists at '{}'. To continue remove it or use '--resume' option.".format(session_path))
        ## Existing "reports_path" is not as severe.
        # if os.path.exists(reports_path):
        #     raise Exception("The 'reports_path' already exists at '{}'. To continue remove it or use '--resume' option.".format(reports_path))

    st = StatLogger(monitor_cpu=True, monitor_gpu=True, output=os.path.join(logs_path, "stats_"+instance+"_"+generateTimestamp()+".log"), interval=2.0)
    st.start()

    session_name_list = []
    session_name_pattern_list = []
    output_dir_list = []
    ###########################
    ##### Run Simulations #####
    ###########################
    command_list = []

    # Different replay steps
    for j in [1, 3, 5, 7, 9]:
        replay_nsteps = j + 1
        for i in [0, 2, 4, 6, 8]:
            replay_use_ratio = (i+1) * 0.1
            replay_use_ratio_str = "{:2.1f}".format(replay_use_ratio).replace(".", "_")
            session_name_seed = "session_{replay_use_ratio_str}_n{replay_nsteps}_{{seed}}".format(replay_use_ratio_str=replay_use_ratio_str, replay_nsteps=replay_nsteps)
            session_name_pattern_list += [session_name_seed.format(seed="s*")]
            output_dir_list += [session_name_seed.format(seed="ALL")]

            for seed in [100, 110, 120]:
                session_name = session_name_seed.format(seed="s"+str(seed))
                command = """python -u -m digideep.main \
                                    --save-modules "dextron" \
                                    --session-path "{session_path:s}" \
                                    --session-name "{session_name:s}" \
                                    --params dextron.params.sac \
                                    --cpanel '{{"time_limit":6, \
                                                "seed":{seed:d}, \
                                                "replay_use_ratio":{replay_use_ratio:2.1f}, \
                                                "replay_nsteps":{replay_nsteps:d}, \
                                                "epoch_size":{epoch_size:d}, \
                                                "number_epochs":{number_epochs:d}\
                                            }}' \
                        """.format(session_path=session_path,
                                session_name=session_name,
                                seed=seed,
                                replay_use_ratio=replay_use_ratio,
                                replay_nsteps=replay_nsteps,
                                epoch_size=400,
                                number_epochs=1000)
                
                command = clean_spaces(command)
                session_name_list += [session_name]
                if not session_name in donelist:
                    command_list += [command]
                # print(command)

    print("Executing simulations.")
    # JobPool(command_list, nproc=22).run().print_err().print_out()
    JobPool(command_list, nproc=None).run().print_all()
    
    ###############################
    ##### Run Post-processing #####
    ###############################
    command_list = []
    for session_name, output_dir in zip(session_name_pattern_list, output_dir_list):
        command = """python -m dextron.post --session-names {} --output-dir {}""".format(session_name, output_dir)
        command = clean_spaces(command)
        command_list += [command]

    JobPool(command_list, nproc=None).run().print_all()

    # #################################
    # ##### Generate plot reports #####
    # #################################
    # #### Copy all of the png files into the parent folder.
    # import ffmpeg # See: https://github.com/kkroening/ffmpeg-python


    if not os.path.exists(reports_path): # And it shouldn't exist indeed.
        os.makedirs(reports_path)
    else:
        print("'reports_path' already exists at '{}'".format(reports_path))

    # for result_mode in ["test", "train"]:
    #     output_path = os.path.join(reports_path, "{}_mosaic.png".format(result_mode))
    #     plot_name = "explore_reward_{}.png".format(result_mode)
    #     # explore_reward_train.png | explore_reward_test.png 
    #     inputs = []
    #     for session_name in session_name_list:
    #         path_to_plot = os.path.join(session_path, session_name, "plots", plot_name)
    #         inputs += [ffmpeg.input(path_to_plot)]
    #     top    = ffmpeg.filter(inputs[0:3], "hstack", inputs="3")
    #     middle = ffmpeg.filter(inputs[3:6], "hstack", inputs="3")
    #     bottom = ffmpeg.filter(inputs[6:9], "hstack", inputs="3")
    #     ffmpeg.filter([top, middle, bottom], "vstack", inputs="3").output(output_path).run(overwrite_output=True)






















## Sample of processing an image sequence to produce a tile of images.
# # Tile a sequence of images
# output_path = os.path.join(reports_path, mode+"_tile.png")
# plot_name = "explore_reward_train.png"
# # session_name = "session_{mode}_*_*".format(mode=mode)
# session_name = "session_{mode}_0_?".format(mode=mode)
# path_to_plot = os.path.join(session_path, session_name, "plots", plot_name)

# stream = ffmpeg.input(path_to_plot, pattern_type='glob')
# stream = stream.filter("scale", w="1200", h="-1")
# stream = stream.filter("tile", "3x3")
# stream.output(output_path).run(overwrite_output=True)




# Video mosaic 3x3
# ffmpeg -i "${FILE}1_pre.mp4" -i "${FILE}2_pre.mp4" -i "${FILE}3_pre.mp4" \
#        -i "${FILE}4_pre.mp4" -i "${FILE}5_pre.mp4" -i "${FILE}6_pre.mp4" \
#        -i "${FILE}7_pre.mp4" -i "${FILE}8_pre.mp4" -i "${FILE}9_pre.mp4" \
#        -filter_complex \
#          "[0:v][1:v][2:v]hstack=inputs=3[top];
#           [3:v][4:v][5:v]hstack=inputs=3[middle];
#           [6:v][7:v][8:v]hstack=inputs=3[bottom];
#           [top][middle][bottom]vstack=inputs=3[v]" -map "[v]" "${FILE}mosaic.mp4"










###########################
##### Generate videos #####
###########################
# Generate videos by playing the checkpoint for a few rounds
# Then, we can concatenate those in a single file.
# We can concatentae enough of them so we have 20s of simulation.
# We can should how it evolves over time.
# Then we can change the format.
# Then 

# ------------------------
# |  1   |   1   |   1   |
# |----------------------|
# |  1   |   1   |   1   |
# |----------------------|
# |  1   |   1   |   1   |
# |----------------------|

# Fade in

# ------------------------
# |  10  |   10  |   10  |
# |----------------------|
# |  10  |   10  |   10  |
# |----------------------|
# |  10  |   10  |   10  |
# |----------------------|

# Fade in

# ------------------------
# | 100  |  100  |  100  |
# |----------------------|
# | 100  |  100  |  100  |
# |----------------------|
# | 100  |  100  |  100  |
# |----------------------|

# Create a set of mosaics of each 20s of a certain fixed checkpoint.
# These mosaics must have labels on figures.
# Also the viewpoint of camera should be a better one.
# Then concatenate them all with fade in effect.


##################################
##### Generate video reports #####
##################################
# Here we set videos in a mosaic format.

# mp4 to avi
# ffmpeg -i "$1.mp4" -q 1 "$1.avi"

# Gen gif
# ffmpeg -i "$1.mp4" -filter_complex "[0:v] fps=12, scale=480:-1 [gif]; [gif] split [a][b]; [a] palettegen [p]; [b][p] paletteuse" "$1.gif"

# Video pre-process
# ffmpeg -ss 0.0 -t 60 -i "$1.mp4" \
#        -filter_complex \
#          "[0:v] setpts=PTS/2, crop=2*in_w/3:2*in_h/3:in_w/6:in_h/6, fps=12, scale=480:-1 [in];
#           [in] drawtext=text='$2':fontsize=36:fontcolor=white@0.8:box=1:boxcolor=black@0.75:boxborderw=12:fontfile=/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-M.ttf:x=w-text_w-12:y=h-text_h-12" \
#         -r 48 -ss 0.0 -t 30 "$1_pre.mp4"

# Video speedup
# BOXBORDERW=24
# FONTSIZE=64
# ffmpeg -i "$1.mp4" \
#        -filter_complex \
#           "[0:v] drawtext=text='$2':fontsize=${FONTSIZE}:fontcolor=white@0.8:box=1:boxcolor=black@0.75:boxborderw=${BOXBORDERW}:fontfile=/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-M.ttf:x=w-text_w-${BOXBORDERW}:y=${BOXBORDERW}" \
#        "$1_speed.mp4"



# Video mosaic 3x3
# ffmpeg -i "${FILE}1_pre.mp4" -i "${FILE}2_pre.mp4" -i "${FILE}3_pre.mp4" \
#        -i "${FILE}4_pre.mp4" -i "${FILE}5_pre.mp4" -i "${FILE}6_pre.mp4" \
#        -i "${FILE}7_pre.mp4" -i "${FILE}8_pre.mp4" -i "${FILE}9_pre.mp4" \
#        -filter_complex \
#          "[0:v][1:v][2:v]hstack=inputs=3[top];
#           [3:v][4:v][5:v]hstack=inputs=3[middle];
#           [6:v][7:v][8:v]hstack=inputs=3[bottom];
#           [top][middle][bottom]vstack=inputs=3[v]" -map "[v]" "${FILE}mosaic.mp4"