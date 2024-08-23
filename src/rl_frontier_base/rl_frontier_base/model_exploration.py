import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class RLFrontierBaseModel(nn.Module):
    def __init__(self):
        super(RLFrontierBaseModel, self).__init__()

        """ 
        Odom and Map Input 
            positionX
            positionY
            YAW(orientationX, orientationY, OrientationZ, orientationW)
            originX
            originy
        """
        self.odom_layer1 = nn.Linear(6, 64)
        self.odom_layer2 = nn.Linear(64, 128)

        """ 
        Map Inputs
            data (normalized for values and sizes as 128x128)
        """
        self.map_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.map_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.map_fc1 = nn.Linear(32 * 128 * 128, 128)

        """
            Combine Layers
        """
        self.combined_fc1 = nn.Linear(128 + 128, 256)
        self.combined_fc2 = nn.Linear(256, 128)

        """ 
        Output Layer 
            left 
            right 
            forward
            back
        """
        self.output = nn.Linear(128, 4)

    def forward(self, odom_and_map_input, map):

        # feed the odom and map layer
        odom_map_x = F.relu(self.odom_layer1(odom_and_map_input))
        odom_map_x = F.relu(self.odom_layer2(odom_map_x))

        # feed map inputs
        map_x = F.relu(self.map_conv1(map))
        map_x = F.relu(self.map_conv2(map_x))
        map_x = map_x.view(-1, 1).t()  # flatten

        map_x = F.relu(self.map_fc1(map_x))

        map_x = map_x.view(1, 128)
        odom_map_x = odom_map_x.view(1, 128)

        # combine odom and map outputs
        combined_x = torch.cat((odom_map_x, map_x), dim=1)
        combined_x = F.relu(self.combined_fc1(combined_x))
        combined_x = F.relu(self.combined_fc2(combined_x))

        # output layer
        output = self.output(combined_x)
        return output

    def save_model(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state0, state1, action, reward, next_state0, next_state1, done):
        state0 = torch.tensor(state0, dtype=torch.float)
        state1 = torch.tensor(state1, dtype=torch.float)

        next_state0 = torch.tensor(next_state0, dtype=torch.float)
        next_state1 = torch.tensor(next_state1, dtype=torch.float)

        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state0.shape) == 1 or len(state1.shape) == 1:
            state0 = torch.unsqueeze(state0, 0)
            state1 = torch.unsqueeze(state1, 0)
            next_state0 = torch.unsqueeze(next_state0, 0)
            next_state1 = torch.unsqueeze(next_state1, 0)

            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # predicted values with curren state
        pred = self.model(state0, state1)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state0[idx], next_state1[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
