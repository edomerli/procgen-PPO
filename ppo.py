import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """IMPALA ConvBlock class. It's a sequence of Conv2d, ReLU and (optionally) BatchNorm2d layers."""
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
    """IMPALA network class. It's a sequence of 'stem' layers (Conv2d + MaxPool2d), residual blocks (two ConvBlocks each) and a fully connected head.
    The output is a categorical distribution if num_actions > 1 (policy network instance), otherwise it's a single value (value network instance).
    The final linear layer is initialized with orthogonal weights and zero bias to help initial exploration.
    """
    def __init__(self, in_channels, num_actions, batch_norm):
        super(ImpalaNetwork, self).__init__()
        
        self.num_actions = num_actions

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

        self.fc = torch.nn.Linear(hidden_channels[-1] * 7 * 7, out_features=256)

        self.out = torch.nn.Linear(256, num_actions)

        
        if num_actions > 1:
            # policy network initialization
            nn.init.orthogonal_(self.fc.weight, gain=0.01)
            nn.init.constant_(self.fc.bias, 0)
        else:
            # value network initialization
            nn.init.orthogonal_(self.out.weight, gain=1)
            nn.init.constant_(self.out.bias, 0)



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
            output = torch.distributions.Categorical(logits=logits)
        else:
            output = self.out(x).squeeze()

        return output

class PPO:
    """PPO agent class. 
    It contains the policy and value networks, and the methods call them and get actions/value estimates.
    """
    def __init__(self, env, config):
        self.policy_net = ImpalaNetwork(config.stack_size * 3, env.action_space.n, config.batch_norm)
        self.value_net = ImpalaNetwork(config.stack_size * 3, 1, config.batch_norm)

        self.normalize_v_targets = config.normalize_v_targets

        if self.normalize_v_targets:
            self.value_mean = 0
            self.value_std = 1
            self.values_count = 0

    # act(), value() and act_and_v() are used during play, hence a single value (.item()) is returned
    def act(self, state):
        dist = self.policy_net(state)
        action = dist.sample()

        return action.item()
    
    def value(self, state):
        value = self.value_net(state)

        if self.normalize_v_targets:
            # denormalize value
            value = value * max(self.value_std, 1e-6) + self.value_mean

        return value.item()

    def act_and_v(self, state):
        action = self.act(state)
        value = self.value(state)

        return action, value
    
    # actions_dist() and actions_dist_and_v() are used during training, hence the full distributions and values are returned
    def actions_dist(self, state):
        return self.policy_net(state)
    
    def actions_dist_and_v(self, state):
        dist = self.policy_net(state)
        value = self.value_net(state)

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
        """If normalize_v_targets is True, will be called to update the mean and std of value targets. This is used to normalize value targets during training."""
        self.value_mean = (self.value_mean * self.values_count + v_targets.mean() * len(v_targets)) / (self.values_count + len(v_targets) + 1e-6)
        self.value_std = (self.value_std * self.values_count + v_targets.std() * len(v_targets)) / (self.values_count + len(v_targets) + 1e-6)
        self.values_count += len(v_targets)

    def state_dict(self):
        return self.policy_net.state_dict()
    
    def load_state_dict(self, state_dict):
        self.policy_net.load_state_dict(state_dict)
