import multiprocessing
import subprocess
import shlex
from multiprocessing.pool import ThreadPool
import os

def call_proc(cmd):
    """ This runs in a separate thread. """
    p = subprocess.Popen(shlex.split(cmd, posix=True), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)

class JobPool:
    def __init__(self, command_list, nproc=None):
        self.nproc = nproc or (multiprocessing.cpu_count()-1)
        self.pool = ThreadPool(self.nproc)
        self.command_list = command_list
        self.results = []

    def run(self):
        for command in self.command_list:
            self.results.append(self.pool.apply_async(call_proc, (command,)))
        # Close the pool and wait for each running task to complete
        self.pool.close()
        self.pool.join()
        return self

    def print_out(self):
        for result, command in zip(self.results, self.command_list):
            out, _ = result.get()
            print("#", command)
            print("."*85)
            print(out.decode("utf-8"))
            print("\n"+"="*85+"\n")
        return self

    def print_err(self):
        for result, command in zip(self.results, self.command_list):
            _, err = result.get()
            print("#", command)
            print("."*85)
            print(err.decode("utf-8"))
            print("\n"+"="*85+"\n")
        return self

    def print_all(self):
        for result, command in zip(self.results, self.command_list):
            out, err = result.get()
            print("#", command)
            print("."*85)
            print(out.decode("utf-8"))
            print("."*85)
            print(err.decode("utf-8"))
            print("\n"+"="*85+"\n")
        return self


session_path = "/tmp/digideep_sessions"
reports_path = "/reports"
session_name_list = []
session_name_pattern_list = []
output_dir_list = []
###########################
##### Run Simulations #####
###########################
command_list = []
mode = "replay"

mode_path = os.path.join(reports_path, mode)
if not os.path.exists(mode_path):
    os.makedirs(mode_path)
else:
    print("Directory '{}' already existed.".format(mode_path))

for i in range(1):
    replay_use_ratio = (i+1) * 0.1
    replay_use_ratio_str = "{:2.1f}".format(replay_use_ratio).replace(".", "_")
    session_name = "session_{mode}_{replay_use_ratio_str}_{{seed}}".format(mode=mode, replay_use_ratio_str=replay_use_ratio_str)
    session_name_pattern_list += [session_name.format(seed="s*")]
    output_dir_list += [session_name.format(seed="ALL")]

    for seed in [100, 110, 120]:
        session_name = session_name.format(seed="s"+str(seed))
        command = """python -m digideep.main \
                            --save-modules "dextron" \
                            --session-name "{session_name:s}" \
                            --params dextron.params.sac \
                            --cpanel '{{"time_limit":6, \
                                        "seed":{seed:d}, \
                                        "replay_use_ratio":{replay_use_ratio:2.1f}, \
                                        "epoch_size":{epoch_size:d}, \
                                        "number_epochs":{number_epochs:d}\
                                    }}' \
                """.format(session_name=session_name,
                           seed=seed,
                           replay_use_ratio=replay_use_ratio,
                           epoch_size=400,
                           number_epochs=1000)
        session_name_list += [session_name]
        command_list += [command]
    
# JobPool(command_list).run().print_err()

###############################
##### Run Post-processing #####
###############################
command_list = []
for session_name, output_dir in zip(session_name_pattern_list, output_dir_list):
    command = """python -m dextron.post --session-names {} --output-dir {}""".format(session_name, output_dir)
    command_list += [command]

JobPool(command_list).run().print_all()

# #################################
# ##### Generate plot reports #####
# #################################
# #### Copy all of the png files into the parent folder.
# import ffmpeg # See: https://github.com/kkroening/ffmpeg-python

# for result_mode in ["test", "train"]:
#     output_path = os.path.join(reports_path, mode, "{}_mosaic.png".format(result_mode))
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