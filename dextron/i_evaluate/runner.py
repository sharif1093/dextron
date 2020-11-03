from digideep.pipeline import Runner
from digideep.utility.profiling import KeepTime
from digideep.utility.logging import logger
import gc, time



class MyRunner (Runner):
    def train(self):
        try:
            while (self.state["i_epoch"] < self.params["runner"]["n_epochs"]) and not self.termination_check():
                self.state["i_cycle"] = 0
                while self.state["i_cycle"] < self.params["runner"]["n_cycles"]:
                    with KeepTime("/"):
                        
                        with KeepTime("demo"):
                            chunk = self.explorer["demo"].update()
                            
                            if self.params["runner"]["is_store"]:
                                with KeepTime("store"):
                                    self.memory["demo"].store(chunk)
                        
                        # if self.memory["demo"].full:
                        #     # Memory full. Time to leave
                        #     self.ready_for_termination
                        #     # Make sure major checkpoints work.

                    self.state["i_cycle"] += 1
                # End of Cycle
                self.state["i_epoch"] += 1
                self.monitor_epoch()
                
                # 3. Log
                self.log()
                gc.collect() # Garbage Collection

        except (KeyboardInterrupt, SystemExit):
            logger.fatal('Operation stopped by the user ...')
        finally:
            logger.fatal('End of operation ...')
            self.finalize()


# if (self.memory["demo"].full and (self.state["i_cycle"] % 3 == 0)) or (not self.memory["demo"].full):
# self._sync_normalizations(source_explorer="train", target_explorer="demo")
# self._sync_normalizations(source_explorer="demo", target_explorer="train")
