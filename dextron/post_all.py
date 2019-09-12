import argparse, os, re

from dextron.dist.jobpool import JobPool
from dextron.dist.utils import clean_spaces

if __name__=="__main__":
    """
    NOTE: For the sessions to be recognized by this code, they need to follow this pattern: "<name>_s<seed>"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root-dir', metavar=('<path>'), default='/tmp/digideep_sessions', type=str, help="The root directory of sessions.")
    args = parser.parse_args()

    SEED_PATTERN = '_s\d+$'

    # 1. Get all directories.
    # 2. Remove any possible s* from the ends of the folder names.
    # 3. Get the unique set of those session names.
    # 4. Iterate over them, add "_s*" to their names to specify the session name and add "sAll" to specify the output dir.
    # 5. Create the command list.
    # 6. Pass it to the JobPool for parallel processing.
    
    sessions_list = [dI for dI in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, dI))]
    
    # Only keep the names that match the SEED PATTERN:
    sessions_list = [dI for dI in sessions_list if re.search(SEED_PATTERN, dI)]
    
    sessions_list_no_seed = [re.sub(SEED_PATTERN, '', dI) for dI in sessions_list]
    sessions_list_no_seed = list(set(sessions_list_no_seed))
    session_name_pattern_list = [dI+"_s*" for dI in sessions_list_no_seed]
    output_dir_list = [dI+"_ALL" for dI in sessions_list_no_seed]

    ###############################
    ##### Run Post-processing #####
    ###############################
    command_list = []
    for session_name, output_dir in zip(session_name_pattern_list, output_dir_list):
        command = """python -m dextron.post --root-dir {} --session-names {} --output-dir {}""".format(args.root_dir, session_name, output_dir)
        command = clean_spaces(command)
        command_list += [command]

    jp = JobPool("Jobs", command_list, nproc=None)
    jp.run().print_all()
