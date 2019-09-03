from .distributed_jobpool import DistributedJobPool
from .job_assignment import partition_greedy
from .utils import get_done_sessions
from .utils import rmdir
# from .utils import pad_matrix

from .jobpool import JobPool

import argparse
import os
# import os, re, glob, shutil, argparse

# import shlex
# import subprocess
# import multiprocessing
# from multiprocessing.pool import ThreadPool

class Executer:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--session-path', metavar=('<path>'), type=str, default="/tmp/digideep_sessions", help="Path to session storage.")
        parser.add_argument('--reports-path', metavar=('<path>'), type=str, default="/reports", help="Path to reports storage.")
        parser.add_argument('--logs-path', metavar=('<path>'), type=str, default="/logs", help="Path to log files.")
        parser.add_argument('--resume', action='store_true', help="Whether to resume already processed data or to remove the existing sessions_path and reports.")
        parser.add_argument('--nproc', metavar=('<n>'), type=int, default=None, help="Number of simultaneous tasks.")
        self.args = parser.parse_args()
        self.dj = DistributedJobPool()
        self.dj.barrier()
    
    def run(self, job_generator):
        session_path = self.args.session_path
        reports_path = self.args.reports_path
        logs_path = self.args.logs_path
        resume = self.args.resume

        # # TODO: Later get it from commandline
        # session_path = "/scratch/sharif.mo/digideep_slurm/sessions/ins_1"
        # reports_path = "/scratch/sharif.mo/digideep_slurm/reports"
        # logs_path = "/scratch/sharif.mo/digideep_slurm/logs"
        self.dj.init(logs_path=logs_path)

        
        # Cleanup the sessions directory by the master node.
        if self.dj.is_master:
            done_sessions, incomplete = get_done_sessions(session_path, resume=resume)
            print("<<<<<<<<<< These sessions are already done >>>>>>>>>>")
            print(done_sessions)
        
            job_list, job_weights = job_generator(session_path, done_sessions)

            # Update job_list based on job_weights.
            dist_job_list, sums, score = partition_greedy(weights=job_weights, n_partitions=self.dj.world_size)
            print("Jobs are divided into {n_partitions:d} groups with a score of {score:.4f}.".format(
                n_partitions=self.dj.world_size,
                score=score))

            task = {"job_list":job_list, "dist_job_list":dist_job_list}
            incomplete = list(incomplete)
        else:
            task = None
            incomplete = None
        
        # Remove incomplete items by all tasks simultaneously:
        incomplete = self.dj.bcast(incomplete, root=0)
        for i in range(self.dj.rank, len(incomplete), self.dj.world_size):
            rmdir(os.path.join(session_path, incomplete[i]))
        
        # Broadcast list of jobs to all nodes.
        task = self.dj.bcast(task, root=0)
        # Each node retrieve its own jobs.    
        # The jobs to be run by 
        command_list = []
        indexes_list = task["dist_job_list"][self.dj.rank]
        for i in indexes_list:
            command_list += [task["job_list"][i]]
        print("Node {nodename:s} ({rank:d}) has {joblength:d} jobs.\n\n".format(nodename=self.dj.nodename, rank=self.dj.rank, joblength=len(command_list)))

        # import time
        # time.sleep(100)
        

        
        ## Each node start its own jobs
        description = "{}_{}({})".format(self.dj.nodename, self.dj.local_rank, self.dj.rank)
        jp = JobPool(description, command_list, nproc=self.args.nproc)
        print("{}: JobPool runs with {} cores.".format(description, jp.nproc))
        jp.run().print_all()
        # print("---------------")
        # print(self.dj.nodename + ":", command_list)

        ## Get number of processes with nproc. Will that be different from multiprocessing.get_processes?
        # JobPool(command_list, nproc=22).run().print_err().print_out()
            
            




        