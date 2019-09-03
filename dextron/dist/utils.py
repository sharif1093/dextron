import os
import re

def clean_spaces(command):
    return re.sub("\s\s+" , " ", command)

def lsdir(root):
    return [x for x in os.listdir(root) if os.path.isdir(os.path.join(root,x))]

def rmdir(path):
    os.system('rm -rf {}'.format(path))

def check_done(root, name):
    return os.path.exists(os.path.join(root, name, 'done.lock'))

# def pad_matrix(matrix, pad_element=-1):
#     s = 0
#     for row in matrix:
#         s = max(len(row), s)
#     for row in matrix:
#         row += [pad_element]*(s-len(row))
#     return matrix

def get_done_sessions(session_path, resume=True):
    if resume and os.path.exists(session_path):
        # Get all directories in the session_path
        # Check if "done.lock" is inside.
        # If "YES", don't do that job again.
        # If "NO", remove that job and ket that to be done again.

        existing = set(lsdir(session_path))
        donesessions = {x for x in existing if check_done(session_path, x)}
        incomplete = existing - donesessions
    else:
        donesessions = {}
        incomplete = {}

        # Complain if the directories already exist.
        # Ask user to remove the directories then continue.
        # This helps increase safety.
        if os.path.exists(session_path):
            raise Exception("The 'session_path' already exists at '{}'. To continue remove it or use '--resume' option.".format(session_path))
        ## Existing "reports_path" is not as severe.
        # if os.path.exists(reports_path):
        #     raise Exception("The 'reports_path' already exists at '{}'. To continue remove it or use '--resume' option.".format(reports_path))
    return donesessions, incomplete
