from .agent_sched import AgentSchedule

from digideep.utility.logging import logger
from digideep.utility.profiling import KeepTime
from digideep.utility.monitoring import monitor

class Agent(AgentSchedule):
    def calculate_loss(self, state, action, reward, next_state, masks):
        with KeepTime("loss"):
            state_repr = self.policy.model["image"](state)
            expected_q_value = self.policy.model["softq"](state_repr.detach(), action)
            expected_value = self.policy.model["value"](state_repr)
            # new_action, log_prob, z, mean, log_std = self.policy.evaluate_actions(state_repr.detach())
            new_action, log_prob, z, mean, log_std = self.policy.evaluate_actions(state_repr.detach())

            next_state_repr = self.policy.model["image"](next_state).detach()
            target_value = self.policy.model["value_target"](next_state_repr)
            next_q_value = reward + masks * float(self.params["methodargs"]["gamma"]) * target_value
            softq_loss = self.criterion["softq"](expected_q_value, next_q_value.detach())

            expected_new_q_value = self.policy.model["softq"](state_repr.detach(), new_action)
            next_value = expected_new_q_value - log_prob
            value_loss = self.criterion["value"](expected_value, next_value.detach())

            log_prob_target = expected_new_q_value - expected_value
            # TODO: Apparently the calculation of actor_loss is problematic: none of its ingredients have gradients! So backprop does nothing.
            actor_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
            
            mean_loss = float(self.params["methodargs"]["mean_lambda"]) * mean.pow(2).mean()
            std_loss  = float(self.params["methodargs"]["std_lambda"])  * log_std.pow(2).mean()
            z_loss    = float(self.params["methodargs"]["z_lambda"])    * z.pow(2).sum(1).mean()

            actor_loss += mean_loss + std_loss + z_loss

        with KeepTime("optimization"):
            self.optimizer["softq"].zero_grad()
            softq_loss.backward()
            self.optimizer["softq"].step()

            self.optimizer["value"].zero_grad()
            value_loss.backward()
            self.optimizer["value"].step()

            # self.optimizer["image"].zero_grad()
            # self.optimizer["image"].step()

            self.optimizer["actor"].zero_grad()
            actor_loss.backward()
            self.optimizer["actor"].step()
            
        with KeepTime("monitor"):
            monitor("/update/loss/actor", actor_loss.item())
            monitor("/update/loss/softq", softq_loss.item())
            monitor("/update/loss/value", value_loss.item())
            