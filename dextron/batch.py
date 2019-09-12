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

    # Different replay steps
    for j in [0, 1, 2]:
        replay_nsteps = j + 1
        for i in [0, 2, 4, 6, 8]:
            replay_use_ratio = (i+1) * 0.1
            replay_use_ratio_str = "{:2.1f}".format(replay_use_ratio).replace(".", "_")
            session_name_seed = "session_{replay_use_ratio_str}_n{replay_nsteps}_{{seed}}".format(replay_use_ratio_str=replay_use_ratio_str, replay_nsteps=replay_nsteps)
            # session_name_pattern_list += [session_name_seed.format(seed="s*")]
            # output_dir_list += [session_name_seed.format(seed="ALL")]

            # Compute an estimate of how much it would take relatively for this task to finish.
            weights = replay_nsteps

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
                if not session_name in done_sessions:
                    command_list += [command]
                    weights_list += [weights]
                # print(command)
    return command_list, weights_list

if __name__=="__main__":
    e = Executer()
    e.run(generate_job_list)
