import threading
import torch
import rclpy
import random
import numpy as np
from collections import deque
from rl_frontier_base.rl_control_similation_node import RLFrontierBaseSimilation
from rl_frontier_base.model_exploration import RLFrontierBaseModel, QTrainer
from helper import plot
import time


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_exploration = 0
        self.epsilon = 0
        self.gamma = 0.9  # discaount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = RLFrontierBaseModel()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def getState(self, similation):
        originX = similation.get_originX()
        originY = similation.get_originY()
        xPosition = similation.get_Xposition()
        yPosition = similation.get_Yposition()
        min_max = similation.get_scan_ranges()  # yaw and others above also should be normalized into some numbers
        data = similation.get_MapData()  # should be normalized and 128x128

        return [
            np.array([originX, originY, xPosition, yPosition, min_max[0], min_max[1]], dtype=int),
            np.array([data], dtype=int),
        ]

    def remember(self, state0, state1, action, reward, next_state0, next_state1, done):
        self.memory.append((state0, state1, action, reward, next_state0, next_state1, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for states0, states1, actions, rewards, next_states0, next_states1, dones in mini_sample:
            self.trainer.train_step(states0, states1, actions, rewards, next_states0, next_states1, dones)

    def train_short_memory(self, state0, state1, action, reward, next_state0, next_state1, done):
        self.trainer.train_step(state0, state1, action, reward, next_state0, next_state1, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_exploration
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            print("RANDOM~")
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            print("AI~")
            state0 = torch.tensor(state[0], dtype=torch.float32)
            state1 = torch.tensor(state[1], dtype=torch.float32)
            prediction = self.model(state0, state1)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def is_similation_launched(self, similation):
        values = []
        values.append(similation.get_originX())
        values.append(similation.get_originY())
        values.append(similation.get_Xposition())
        values.append(similation.get_Yposition())
        values.append(similation.get_scan_ranges())  # yaw and others above also should be normalized into some numbers
        values.append(similation.get_MapData())  # should be normalized and 128x128

        # Check if any value in the list is None
        return not any(value is None for value in values)


def spin_thread(node):
    rclpy.spin(node)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    exp_record = 0
    agent = Agent()

    max_iteration_to_kill = 0

    rclpy.init()
    similation = RLFrontierBaseSimilation()

    print("-----------------------------------------------------------------")
    thread = threading.Thread(target=spin_thread, args=(similation,))
    thread.start()

    while True:

        if not agent.is_similation_launched(similation):
            continue

        print("Similation is running")
        max_iteration_to_kill += 1
        state_old = agent.getState(similation)

        final_move = agent.get_action(state_old)

        reward, done, exp_score = similation.move_step(final_move)

        state_new = agent.getState(similation)

        agent.train_short_memory(state_old[0], state_old[1], final_move, reward, state_new[0], state_new[1], done)

        agent.remember(state_old[0], state_old[1], final_move, reward, state_new[0], state_new[1], done)
        print("REWARD---", reward)

        if done:
            similation.reset()
            agent.n_exploration += 1
            agent.train_long_memory()

            if exp_score > exp_record:
                exp_record = exp_score
                agent.model.save_model()

            print("Similation:", agent.n_exploration, "Exp Score:", exp_score, "Exp Record:", exp_record)

            plot_scores.append(exp_score)
            total_score += exp_score
            mean_score = total_score / agent.n_exploration
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

        if max_iteration_to_kill > 1000:
            break

    thread.join()  # Bu, ana programın spin thread'in tamamlanmasını beklemesini sağlar.
    similation.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    train()
