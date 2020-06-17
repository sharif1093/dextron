from digideep.pipeline import Runner
from digideep.utility.profiling import KeepTime
from digideep.utility.logging import logger
import gc, time


class RandomRunner (Runner):
    def train(self):
        try:
            while self.state["i_epoch"] < self.params["runner"]["n_epochs"]:
                self.state["i_cycle"] = 0
                while self.state["i_cycle"] < self.params["runner"]["n_cycles"]:
                    with KeepTime("/"):
                        self.explorer["demo"].update()

                        # # Update Agent
                        # for agent_name in self.agents:
                        #     with KeepTime(agent_name):
                        #         self.agents[agent_name].update()

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
