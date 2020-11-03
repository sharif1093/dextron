from digideep.pipeline import Runner
from digideep.utility.profiling import KeepTime
from digideep.utility.logging import logger
import gc, time

class StoreRunner (Runner):
    def train(self):
        try:
            self.explorer["train"].reset()

            while (self.state["i_epoch"] < self.params["runner"]["n_epochs"]) and not self.termination_check():
                self.state["i_cycle"] = 0
                while self.state["i_cycle"] < self.params["runner"]["n_cycles"]:
                    with KeepTime("/"):
                        with KeepTime("train"):
                            chunk = self.explorer["train"].update()
                            with KeepTime("store"):
                                self.memory["train"].store(chunk)

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
