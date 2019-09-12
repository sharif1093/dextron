# For help: https://pytorch.org/docs/stable/distributed.html
import os, socket
# import torch
# import torch.distributed as dist
# see: https://mpi4py.readthedocs.io/en/stable/tutorial.html

try:
    from mpi4py import MPI
    dist = MPI.COMM_WORLD
except:
    dist = None

import os, re, glob, shutil, argparse

from digideep.utility.stats import StatLogger
from digideep.pipeline.session import generateTimestamp

class DistributedJobPool:
    def __init__(self, backend="gloo"):
        self.master_addr = os.environ.get('MASTER_ADDR', None)
        self.master_port = os.environ.get('MASTER_PORT', None)
        self.world_size  = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        self.nodename = socket.gethostname() # os.environ["SLURMD_NODENAME"]
        # self.setup(backend)

        print("Node ({rank:d}/{world_size:d}) was initialized on '{nodename:s}' with local rank ({local_rank:d}).".format(
            rank=self.rank,
            world_size=self.world_size,
            nodename=self.nodename,
            local_rank=self.local_rank
        ))
    
    @property
    def is_master(self):
        return self.rank == 0
    
    def barrier(self):
        if self.world_size > 1:
            dist.Barrier
    def bcast(self, data, root=0):
        if self.world_size > 1:
            return dist.bcast(data, root=root)
        else:
            return data

    def setup(self, backend):
        pass
        # if self.world_size:
        #     dist.init_process_group(backend=backend, init_method="env://")
        # backend=nccl|gloo|mpi
        # rank=rank, world_size=world_size
        # torch.manual_seed(42)
    def init(self, logs_path):
        # initialize the stat logger
        # Start the resource logger in all nodes.
        # Only those runner must run this that have local_rank == 0
        if self.local_rank == 0:
            output = os.path.join(logs_path, "stats_"+self.nodename+"_"+generateTimestamp()+".log")
            st = StatLogger(monitor_cpu=True, monitor_gpu=True, output=output, interval=10.0)
            st.start()
            print("Stat server was started on node [{:s}] by rank [{:d}].".format(self.nodename, self.rank))
            print("Logs will be stored in '{:s}'.".format(output))
        

    # def cleanup(self):
    #     if self.world_size:
    #         dist.destroy_process_group()
    



# def run(rank, world_size):
#     """ When using `isend`, the number of `wait` commands should match the `isend`'s.
#     If program hang it means some nodes are not finishing their jobs, i.e. waiting at
#     some point, etc.
#     """
#     tensor = torch.zeros(1)
#     req = None
#     group = dist.new_group([0,1])
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         req = dist.send(tensor=tensor, dst=1)
#         req = dist.send(tensor=tensor+1, dst=2)
#         req = dist.send(tensor=tensor+4, dst=3)
#         req = dist.send(tensor=tensor-10, dst=4)
#         req = dist.send(tensor=tensor+20, dst=5)
#         # req.wait()
#         # print('Rank {} started sending'.format(rank))
#     else:
#         # Receive tensor from process 0
#         req = dist.irecv(tensor=tensor, src=0)
#         # print('Rank {} started receiving'.format(rank))
#         req.wait()
#         print('Rank ', rank, ' has data ', tensor[0])


