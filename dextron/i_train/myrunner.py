import gc
import sys
import time
import signal

from digideep.environment import Explorer
from digideep.utility.logging import logger
from digideep.utility.toolbox import seed_all, get_class, get_module, set_rng_state, get_rng_state
from digideep.utility.profiling import profiler, KeepTime
from digideep.utility.monitoring import monitor
from collections import OrderedDict as odict

# Runner should be irrelevant of torch, gym, dm_control, etc.

class Runner:
    def __init__(self, params):
        self.params = params
        self.state = {}
        self.state["i_frame"] = 0
        # self.state["i_rolls"] = 0
        self.state["i_cycle"] = 0
        self.state["i_epoch"] = 0
        self.state["loading"] = False

    def lazy_connect_signal(self):
        # Connect shell signals
        signal.signal(signal.SIGUSR1, self.on_sigusr1_received)
        signal.signal(signal.SIGINT,  self.on_sigint_received)

    def on_sigint_received(self, signalNumber, frame):
        print("") # To print on the next line where ^C is printed.
        self.ctrl_c_count += 1
        if self.ctrl_c_count == 1:
            logger.fatal("Received CTRL+C. Will terminate process after cycle is over.")
            logger.fatal("Press CTRL+C one more time to exit without saving.")
            self.ready_for_termination = True
            self.save_major_checkpoint = True
        elif self.ctrl_c_count == 2:
            # NOTE: Kill all subprocesses
            logger.fatal("Received CTRL+C for the second time. Will terminate immediately.")
            self.ready_for_termination = True
            self.save_major_checkpoint = False
            sys.exit(1)
    
    def on_sigusr1_received(self, signalNumber, frame):
        logger.fatal("Received SIGUSR1 signal. Will terminate process after cycle is over.")
        self.ready_for_termination = True
        self.save_major_checkpoint = True


    def lazy_init(self):
        """
        Initialization of attributes which are not part of the object state.
        These need lazy initialization due to proper initialization when loading
        from a checkpoint.
        """
        self.time_start = time.time()
        logger.fatal("Execution (max) timer started ...")

        self.save_major_checkpoint = False
        self.ready_for_termination = False
        self.iterations = 0

        profiler.reset()
        monitor.reset()
        self.monitor_epoch()

        # Ignore interrupt signals for_subprocesses
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.ctrl_c_count = 0

    def monitor_epoch(self):
        monitor.set_meta_key("epoch", self.state["i_epoch"])
    

    def start(self, session):
        """A function to initialize the objects and load their states (if loading from a checkpoint).
        This function must be called before using the :func:`train` and :func:`enjoy` functions.

        If we are starting from scrarch, we will:

        * Instantiate all internal components using parameters.

        If we are loading from a saved checkpoint, we will:

        * Instantiate all internal components using old parameters.
        * Load all state dicts.
        * (OPTIONAL) Override parameters.
        """
        # Up to now, states of the runner are already loaded. Objects' states not, however.
        self.lazy_init()
        self.session = session
        
        seed_all(**self.params["runner"]["randargs"])

        # The order is as it is:
        self.instantiate()
        self.load()
        self.override()

        # NOTE: We lazily connect signals so it is not spawned in the child processes.
        self.lazy_connect_signal()

        # NOTE: We set this state for the future.
        #       Because all future loading would
        #       involve actual loading of states.
        self.state["loading"] = True

    def instantiate(self):
        """
        This function will instantiate the memory, the explorers, and the agents with their specific parameters.
        """
        ## Instantiate Memory
        self.memory = {}
        for memory_name in self.params["memory"]:
            memory_class = get_class(self.params["memory"][memory_name]["type"])
            self.memory[memory_name] = memory_class(self.session, mode=memory_name, **self.params["memory"][memory_name]["args"])

    ###############################################################
    ### SERIALIZATION ###
    #####################
    def state_dict(self):
        """
        This function will return the states of all internal objects:

        * Agents
        * Explorer (only the ``train`` mode)
        * Memory

        Todo:
            Memory should be dumped in a separate file, since it can get really large.
            Moreover, it should be optional.
        """
        random_state = get_rng_state()

        memory_state = {}
        for memory_name in self.memory:
            memory_state[memory_name] = self.memory[memory_name].state_dict()
        
        return {'random_state':random_state, 'memory':memory_state}
    
    def load_state_dict(self, state_dict):
        """
        This function will load the states of the internal objects:

        * Agents
        * Explorers (state of ``train`` mode would be loaded for ``test`` and ``eval`` as well)
        * Memory
        """
        random_state = state_dict['random_state']
        set_rng_state(random_state)

        memory_state = state_dict['memory']
        for memory_name in memory_state:
            self.memory[memory_name].load_state_dict(memory_state[memory_name])

    def override(self):
        pass

    #####################
    ###  SAVE RUNNER  ###
    #####################
    # UPON SAVING/LOADING THE RUNNER WITH THE SELF.SAVE FUNCTION:
    #   * save --> self.state_dict --> session.save_states --> torch.save --> states.pt
    #          |-> session.save_runner --> self.__getstate__ --> pickle.dump --> runner.pt
    #   * pickle.load --> __setstate__ 
    #     ... Later on ...
    #     --> self.start --> self.instantiate --> self.load --> session.load_states --> self.load_state_dict --> self.override
    # The __setstate__ and __getstate__ functions are for loading/saving the "runner" through pickle.dump / pickle.load
    # 
    def __getstate__(self):
        """
        This function is used by ``pickle.dump`` when we save the :class:`Runner`.
        This saves the ``params`` and ``state`` of the runner.
        """
        # This is at the time of pickling
        state = {'params':self.params, 'state':self.state}
        return state
    def __setstate__(self, state):
        """
        This function is used by ``pickle.load`` when we load the :class:`Runner`.
        """
        # state['state']['loading'] = True
        self.__dict__.update(state)
    ###
    def save_final_checkpoint(self):
        self.save(forced=True)
        # Store snapshots for all memories only if simulation ended gracefully.
        for memory_name in self.memory:
            if hasattr(self.memory[memory_name], "save_snapshot"):
                self.memory[memory_name].save_snapshot(self.state["i_epoch"])

    def save(self, forced=False):
        """
        This is a high-level function for saving both the state of objects and the runner object.
        It will use helper functions from :class:`~digideep.pipeline.session.Session`.
        """
        if forced or (self.state["i_epoch"] % self.params["runner"]["save_int"] == 0):
            ## 1. state_dict: Saved with torch.save
            self.session.save_states(self.state_dict(), self.state["i_epoch"])
            ## 2. runner: Saved with pickle.dump
            self.session.save_runner(self, self.state["i_epoch"])
    def load(self): # This function does not directly work with files. Instead, it 
        """
        This is a function used by the :func:`start` function to load the states of internal objects 
        from the checkpoint and update the objects state dicts.
        """
        if self.state["loading"]:
            state_dict = self.session.load_states()
            self.load_state_dict(state_dict)
            self.load_memory()
            # We leave loading = True. All future loadings would be either resume or play.

    def load_memory(self):
        if self.session.is_resumed:
            for memory_name in self.memory:
                if hasattr(self.memory[memory_name], "load_snapshot"):
                    self.memory[memory_name].load_snapshot()
    ###############################################################

    def train_cycle(self):
        chunk = self.explorer["train"].update()
        self.memory["train"].store(chunk)
        

    def train(self):
        try:
            while (self.state["i_epoch"] < self.params["runner"]["n_epochs"]) and not self.termination_check():
                self.state["i_cycle"] = 0
                while self.state["i_cycle"] < self.params["runner"]["n_cycles"]:
                    with KeepTime("/"):
                        self.train_cycle()
                    self.state["i_cycle"] += 1
                    # End of Cycle
                self.state["i_epoch"] += 1
                self.monitor_epoch()
                self.iterations += 1
                
                # NOTE: We may save/test after each cycle or at intervals.
                # 1. Perform the test
                self.test()
                # 2. Log
                self.log()
                # 3. Save
                self.save()
                # Free up memory from garbage.
                gc.collect() # Garbage Collection

        except (KeyboardInterrupt, SystemExit):
            logger.fatal('Operation stopped by the user ...')
        finally:
            self.finalize()

    def termination_check(self):
        termination = self.ready_for_termination
        if self.params["runner"]["max_time"]:
            if time.time() - self.time_start >= self.params["runner"]["max_time"] * 3600:
                self.save_major_checkpoint = True
                termination = True
                logger.fatal('Simulation maximum allowed execution time exceeded ...')
        if self.params["runner"]["max_iter"]:
            # TODO: Should be current_epoch - initial_epoch >= max_iter: ...
            if self.iterations >= self.params["runner"]["max_iter"]:
                self.save_major_checkpoint = True
                termination = True
                logger.fatal('Simulation maximum allowed execution iterations exceeded ...')
        return termination
    

    def finalize(self, save=True):
        logger.fatal('End of operation ...')
        
        # Mark session as done if we have went through all epochs.
        # if self.state["i_epoch"] == self.state["n_epochs"]:
        if self.state["i_epoch"] == self.params["runner"]["n_epochs"]:
            self.session.mark_as_done()
            self.save_major_checkpoint = True
        
        if save and self.save_major_checkpoint:
            self.save_final_checkpoint()
            # self.save_major_checkpoint = False
        
        # Close all explorers benignly:
        for key in self.explorer:
            self.explorer[key].close()


    def test(self):
        # Make the states of the two explorers train/test exactly the same, for the states of the environments.
        if self.params["runner"]["test_act"]:
            if self.state["i_epoch"] % self.params["runner"]["test_int"] == 0:
                with KeepTime("/"):
                    with KeepTime("test"):
                        self._sync_normalizations(source_explorer="train", target_explorer="test")
                        # self.explorer["test"].load_state_dict(self.explorer["train"].state_dict())
                        self.explorer["test"].reset()
                        # TODO: Do update until "win_size" episodes get executed.
                        # That is in: self.explorer["test"].state["n_episode"]
                        # Make sure that n_steps is 1.
                        # If num_worker>1 it is possible that we get more than required test episodes.
                        # The rest will be reported with the next test run.
                        self.explorer["test"].update()

    def enjoy(self): #i.e. eval
        """This function evaluates the current policy in the environment. It only runs the explorer in a loop.

        .. code-block:: python

            # Do a cycle
            while not done:
                # Explore
                explorer["eval"].update()

            log()
        """
        try:
            self._sync_normalizations(source_explorer="train", target_explorer="eval")
            self.explorer["eval"].reset()
            while True:
                # Cycles
                self.state["i_cycle"] = 0
                while self.state["i_cycle"] < self.params["runner"]["n_cycles"]:
                    with KeepTime("/"):
                        # 1. Do Experiment
                        with KeepTime("eval"):
                            self.explorer["eval"].update()
                    self.log()
                    self.state["i_cycle"] += 1
                # Log
        except (KeyboardInterrupt, SystemExit):
            logger.fatal('Operation stopped by the user ...')
        finally:
            self.finalize(save=False)


    #####################
    def _sync_normalizations(self, source_explorer, target_explorer):
        state_dict = self.explorer[source_explorer].state_dict()
        # digideep.environment.wrappers.normalizers:VecNormalizeObsDict         Observation Normalization States
        # digideep.environment.wrappers.normalizers:VecNormalizeRew             Reward Normalizing States
        # digideep.environment.wrappers.random_state:VecRandomState             Random Generator States
        # digideep.environment.common.vec_env.subproc_vec_env:SubprocVecEnv     Physical states
        
        keys = ["digideep.environment.wrappers.normalizers:VecNormalizeObsDict", "digideep.environment.wrappers.normalizers:VecNormalizeRew"]
        
        state_dict_mod = {}
        for k in keys:
            if k in state_dict["envs"]:
                state_dict_mod[k] = state_dict["envs"][k]

        self.explorer[target_explorer].envs.load_state_dict(state_dict_mod)

    #####################
    ## Logging Summary ##
    #####################
    def log(self):
        """ The log function prints a summary of:

        * Frame rate and simulated frames.
        * Variables sent to the :class:`~digideep.utility.monitoring.Monitor`.
        * Profiling information, i.e. registered timing information in the :class:`~digideep.utility.profiling.Profiler`.
        """
        # monitor.get_meta_key("frame")
        # monitor.get_meta_key("episode")
        # monitor.get_meta_key("epoch")
        
        frame = monitor.get_meta_key("frame")
        episode = monitor.get_meta_key("episode")
        
        n_frame = frame - self.state["i_frame"]
        self.state["i_frame"] = frame
        elapsed = profiler.get_time_overall("/")
        overall = int(n_frame / elapsed)
        
        logger("---------------------------------------------------------")
        logger("Epoch({cycle:3d}cy)={epoch:4d} | Frame={frame:4.1e} | Episodes={episode:4.1e} | Overall({n_frame:4.1e}F/{e_time:4.1f}s)={freq:4d}Hz".format(
                cycle=self.params["runner"]["n_cycles"],
                epoch=self.state["i_epoch"],
                frame=frame,
                episode=episode,
                n_frame=n_frame,
                e_time=elapsed,
                freq=overall
                )
            )
        
        # Printing monitoring information:
        logger("MONITORING:\n"+str(monitor))
        monitor.dump()
        monitor.reset()

        # Printing profiling information:
        logger("PROFILING:\n"+str(profiler))
        meta = odict({"epoch":self.state["i_epoch"],
                      "frame":frame,
                      "episode":episode})

        profiler.dump(meta)
        profiler.reset()

        print("")
