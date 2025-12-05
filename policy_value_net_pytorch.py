# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch using ResNet architecture
Optimized for Gomoku with configurable network parameters

@author: Junxiao Song (original), enhanced with ResNet
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.core import multiarray
from torch.autograd import Variable
from torch.serialization import add_safe_globals

add_safe_globals([multiarray._reconstruct])


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        # Use broadcasting instead of expand_as for efficiency
        return x * y


class ResidualBlock(nn.Module):
    """Enhanced residual block with SE attention and optional dropout"""
    def __init__(self, channels, use_se=True, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Add SE block for channel attention
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE attention
        if self.use_se:
            out = self.se(out)
        
        # Apply dropout if configured
        if self.dropout is not None:
            out = self.dropout(out)
        
        out += residual  # skip connection
        out = F.relu(out)
        return out


class Net(nn.Module):
    """Enhanced ResNet-based policy-value network for AlphaZero with SE blocks"""
    def __init__(self, board_width, board_height, num_channels=128, num_res_blocks=6, 
                 use_se=True, dropout_rate=0.0):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.num_channels = num_channels

        # Initial convolution block
        self.conv_initial = nn.Conv2d(4, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(num_channels)

        # Residual tower with optional SE blocks and dropout
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels, use_se=use_se, dropout_rate=dropout_rate) 
            for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 4, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_width * board_height, 128)
        self.value_fc2 = nn.Linear(128, 64)
        self.value_fc3 = nn.Linear(64, 1)

    def forward(self, state_input):
        # Initial block
        x = F.relu(self.bn_initial(self.conv_initial(state_input)))

        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 4 * self.board_width * self.board_height)
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 2 * self.board_width * self.board_height)
        v = F.relu(self.value_fc1(v))
        v = F.relu(self.value_fc2(v))
        v = torch.tanh(self.value_fc3(v))

        return p, v


class PolicyValueNet():
    """Enhanced policy-value network with optimization options"""
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False,
                 num_channels=128, num_res_blocks=6,
                 use_se=True, dropout_rate=0.0, l2_const=1e-4):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = l2_const  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(
                board_width, board_height,
                num_channels=num_channels,
                num_res_blocks=num_res_blocks,
                use_se=use_se,
                dropout_rate=dropout_rate
            ).cuda()
        else:
            self.policy_value_net = Net(
                board_width, board_height,
                num_channels=num_channels,
                num_res_blocks=num_res_blocks,
                use_se=use_se,
                dropout_rate=dropout_rate
            )
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            map_location = "cuda" if self.use_gpu else "cpu"
            try:
                net_params = torch.load(model_file, map_location=map_location, weights_only=False)
            except UnicodeDecodeError:
                net_params = torch.load(model_file, map_location=map_location, encoding="latin1", weights_only=False)
            except TypeError:
                try:
                    net_params = torch.load(model_file, map_location=map_location)
                except UnicodeDecodeError:
                    net_params = torch.load(model_file, map_location=map_location, encoding="latin1")
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_np = np.array(state_batch)
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_np).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        state_batch = Variable(torch.FloatTensor(state_np))
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.numpy())
        return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
            value = value.data.cpu().numpy()[0][0]
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
            value = value.data.numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr, grad_clip=None):
        """Perform a training step with optional gradient clipping"""
        # wrap in Variable
        state_np = np.array(state_batch)
        mcts_probs_np = np.array(mcts_probs)
        winner_np = np.array(winner_batch)
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_np).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs_np).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_np).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_np))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs_np))
            winner_batch = Variable(torch.FloatTensor(winner_np))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        
        # Apply gradient clipping if specified
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), grad_clip)
        
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
