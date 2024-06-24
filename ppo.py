
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm):
        super(ConvBlock, self).__init__()

        if batch_norm:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            )

    def forward(self, x):
        return self.layer(x)

class ImpalaNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_actions, batch_norm):
        super(ImpalaNetwork, self).__init__()
        
        self.num_actions = num_actions
        # TODO: try flag to add batchnorm? If so, put it always after Conv2d and before ReLU (or after ReLU, check IPCV)

        self.stems = nn.ModuleList()
        self.res_blocks1 = nn.ModuleList()
        self.res_blocks2 = nn.ModuleList()

        hidden_channels = [16, 32, 32]

        for out_channels in hidden_channels:

            # Don't use batch_norm in the first layer as it should go after MaxPool2d, 
            # but it's already present in the successive ConvBlock
            self.stems.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same"),
                torch.nn.MaxPool2d(kernel_size=3, stride=2)
            ))

            self.res_blocks1.append(torch.nn.Sequential(
                ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same", batch_norm=batch_norm),
                ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same", batch_norm=batch_norm),
            ))

            self.res_blocks2.append(torch.nn.Sequential(
                ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same", batch_norm=batch_norm),
                ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same", batch_norm=batch_norm),
            ))

            in_channels = out_channels

        # TODO: do it without magic number 1568 = 7 * 7 * 32
        self.fc = torch.nn.Linear(1568, out_features=256)

        self.out = torch.nn.Linear(256, num_actions)

        
        if num_actions > 1:
            # TODO: remove!
            # # policy network initialization
            # for stem, res_block1, res_block2 in zip(self.stems, self.res_blocks1, self.res_blocks2):
            #     # stem initialization
            #     nn.init.orthogonal_(stem[0].weight, gain=np.sqrt(2))
            #     nn.init.constant_(stem[0].bias, 0)

            #     # res_block1 initialization
            #     for layer in res_block1:
            #         if isinstance(layer, ConvBlock):
            #             nn.init.orthogonal_(layer.layer[2].weight, gain=np.sqrt(2))
            #             nn.init.constant_(layer.layer[2].bias, 0)

            #     # res_block2 initialization
            #     for layer in res_block2:
            #         if isinstance(layer, ConvBlock):
            #             nn.init.orthogonal_(layer.layer[2].weight, gain=np.sqrt(2))
            #             nn.init.constant_(layer.layer[2].bias, 0)

            # policy network initialization
            nn.init.orthogonal_(self.fc.weight, gain=0.01)
            nn.init.constant_(self.fc.bias, 0)
        else:
            # value network initialization
            nn.init.orthogonal_(self.out.weight, gain=1)
            nn.init.constant_(self.out.bias, 0)

        # TODO: convert all network to double in order to avoid softmax <0 / inf / nan problems!


    def forward(self, x):
        for stem, res_block1, res_block2 in zip(self.stems, self.res_blocks1, self.res_blocks2):
            x = stem(x)
            x = res_block1(x) + x
            x = res_block2(x) + x

        x = nn.functional.relu(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = nn.functional.relu(x)

        if self.num_actions > 1:
            logits = self.out(x)
            # TODO: clip the logits between like -20 and 20 to avoid overflow in LogSumExp of Categorical dist!
            output = torch.distributions.Categorical(logits=logits)
        else:
            output = self.out(x).squeeze()

        return output

class PPO:
    def __init__(self, env, config):
        self.policy_net = ImpalaNetwork(config.stack_size * 3, env.action_space.n, config.batch_norm)
        self.value_net = ImpalaNetwork(config.stack_size * 3, 1, config.batch_norm)

        self.normalize_v_targets = config.normalize_v_targets

        if self.normalize_v_targets:
            self.value_mean = 0
            self.value_std = 1
            self.values_count = 0

    def act(self, state):
        dist, value = self.actions_dist_and_v(state)
        action = dist.sample()

        return action.item(), value.item()
    
    def actions_dist_and_v(self, state):
        dist = self.policy_net(state)
        value = self.value_net(state)

        if self.normalize_v_targets:
            # denormalize value
            value = value * max(self.value_std, 1e-6) + self.value_mean

        return dist, value
      
    def to(self, device):
        self.policy_net.to(device)
        self.value_net.to(device)

    def eval(self):
        self.policy_net.eval()
        self.value_net.eval()

    def train(self):
        self.policy_net.train()
        self.value_net.train()

    def update_v_target_stats(self, v_targets):
        self.value_mean = (self.value_mean * self.values_count + v_targets.mean() * len(v_targets)) / (self.values_count + len(v_targets) + 1e-6)
        self.value_std = (self.value_std * self.values_count + v_targets.std() * len(v_targets)) / (self.values_count + len(v_targets) + 1e-6)
        self.values_count += len(v_targets)