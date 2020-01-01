from dextron.dist.exec import Executer
from dextron.dist.utils import clean_spaces

def generate_job_list(session_path, done_sessions):
    session_name_list = []
    session_name_pattern_list = []
    output_dir_list = []
    ###########################
    ##### Run Simulations #####
    ###########################
    command_list = []
    weights_list = []

    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        demo_use_ratio = (i+1) * 0.1
        demo_use_ratio_str = "{:2.1f}".format(demo_use_ratio).replace(".", "_")
        session_name_seed = "session_{demo_use_ratio_str}_{{seed}}".format(demo_use_ratio_str=demo_use_ratio_str)

        # Compute an estimate of how much it would take relatively for this task to finish.
        weights = 1

        for seed in [100, 110, 120]:
            session_name = session_name_seed.format(seed="s"+str(seed))
            command = """python -u -m digideep.main \
                                --save-modules "dextron" \
                                --session-path "{session_path:s}" \
                                --session-name "{session_name:s}" \
                                --params dextron.params.sac \
                                --cpanel '{{"time_limit":6, \
                                            "seed":{seed:d}, \
                                            "demo_use_ratio":{demo_use_ratio:2.1f}, \
                                            "epoch_size":{epoch_size:d}, \
                                            "number_epochs":{number_epochs:d}\
                                        }}' \
                    """.format(session_path=session_path,
                            session_name=session_name,
                            seed=seed,
                            demo_use_ratio=demo_use_ratio,
                            epoch_size=400,
                            number_epochs=1000)
            
            command = clean_spaces(command)
            session_name_list += [session_name]
            if not session_name in done_sessions:
                command_list += [command]
                weights_list += [weights]
            # print(command)
    return command_list, weights_list

if __name__=="__main__":
    e = Executer()
    e.run(generate_job_list)
