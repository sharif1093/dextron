import shlex
import subprocess
# import multiprocessing
from multiprocessing.pool import Pool
from tqdm import tqdm
# from multiprocessing.pool import ThreadPool
from digideep.utility.stats.cpu import get_cpu_count

########################
### Subprocess stuff ###
########################
def call_proc(cmd):
    """ This runs in a separate thread. """
    # print("cmd>", cmd)
    p = subprocess.Popen(shlex.split(cmd, posix=True), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)

class JobPool:
    def __init__(self, description, command_list, nproc=None):
        self.description = description
        self.nproc = nproc or max(get_cpu_count()-2, 1)
        self.pool = Pool(self.nproc)
        self.command_list = command_list
        self.results = []

    def run(self):
        pbar = tqdm(total=len(self.command_list), desc=self.description)
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