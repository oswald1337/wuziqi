import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game import Board
from tactical import ranked_tactical_moves

import logging

logger = logging.getLogger(__name__)

HEURISTIC_PRIOR_WEIGHT = 0.35

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ConvAttentionBlock(nn.Module):
    def __init__(self, num_filters, reduction=4):
        super(ConvAttentionBlock, self).__init__()
        hidden = max(4, num_filters // reduction)
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_filters, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_filters, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.channel_attention(out)
        avg_map = torch.mean(out, dim=1, keepdim=True)
        max_map, _ = torch.max(out, dim=1, keepdim=True)
        out = out * self.spatial_attention(torch.cat([avg_map, max_map], dim=1))
        out += residual
        out = F.relu(out)
        return out


class GomokuNet(nn.Module):
    def __init__(
        self,
        board_width=15,
        board_height=15,
        num_res_blocks=4,
        num_filters=64,
        architecture="residual",
    ):
        super(GomokuNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.architecture = architecture

        # Input: 4 channels (Player stones, Opponent stones, Last move, Color to play)
        self.conv_input = nn.Conv2d(4, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)

        if architecture == "residual":
            block_cls = ResidualBlock
        elif architecture == "conv_attention":
            block_cls = ConvAttentionBlock
        else:
            raise ValueError(f"Unknown model architecture: {architecture}")

        # Residual Tower
        self.res_blocks = nn.ModuleList([
            block_cls(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 4, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch_size, 4, board_width, board_height)
        out = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_blocks:
            out = block(out)

        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(-1, 4 * self.board_width * self.board_height)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(-1, 2 * self.board_width * self.board_height)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

class PolicyValueNet:
    """
    The policy-value network wrapper
    """
    def __init__(
        self,
        board_width=15,
        board_height=15,
        model_file=None,
        use_gpu=True,
        num_res_blocks=4,
        num_filters=64,
        architecture="residual",
    ):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.architecture = architecture
        self.optimizer = None
        self.last_train_components = {}
        
        mps_backend = getattr(torch.backends, "mps", None)
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"PolicyValueNet initialized on CUDA: {torch.cuda.get_device_name(0)}")
        elif self.use_gpu and mps_backend is not None and mps_backend.is_available():
            self.device = torch.device("mps")
            logger.info("PolicyValueNet initialized on Apple MPS")
        else:
            self.device = torch.device("cpu")
            logger.info("PolicyValueNet initialized on CPU")

        self.policy_value_net = GomokuNet(
            board_width,
            board_height,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            architecture=architecture,
        ).to(self.device)

        if model_file:
            net_params = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_params)

    def _optimizer_for_lr(self, lr):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.policy_value_net.parameters(),
                lr=lr,
                weight_decay=self.l2_const,
            )
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        return self.optimizer

    def load_optimizer(self, optimizer_file, lr):
        optimizer = self._optimizer_for_lr(lr)
        optimizer_state = torch.load(optimizer_file, map_location=self.device)
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
        for group in optimizer.param_groups:
            group["lr"] = lr

    def save_optimizer(self, optimizer_file):
        if self.optimizer is None:
            return False
        torch.save(self.optimizer.state_dict(), optimizer_file)
        return True

    def _heuristic_prior(self, board):
        legal_positions = list(board.availables)
        if not legal_positions:
            return np.array([])

        ranked = ranked_tactical_moves(board)
        scores_by_move = {item["move"]: item["score"] for item in ranked}
        scores = [scores_by_move.get(move, 0.0) for move in legal_positions]

        scores = np.array(scores, dtype=np.float64)
        if np.max(scores) > 50_000:
            scores = np.exp((scores - np.max(scores)) / 20_000.0)
        else:
            scores = np.exp((scores - np.max(scores)) / 60.0)
        scores_sum = np.sum(scores)
        if scores_sum <= 0:
            return np.full(len(legal_positions), 1.0 / len(legal_positions))
        return scores / scores_sum

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        
        self.policy_value_net.eval()
        with torch.no_grad():
            if self.use_gpu:
                log_act_probs, value = self.policy_value_net(
                        torch.from_numpy(current_state).float().to(self.device))
                act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
                value = value.data.cpu().numpy()[0][0]
            else:
                log_act_probs, value = self.policy_value_net(
                        torch.from_numpy(current_state).float())
                act_probs = np.exp(log_act_probs.data.numpy().flatten())
                value = value.data.numpy()[0][0]
            
        legal_probs = act_probs[legal_positions]
        prior = self._heuristic_prior(board)
        if len(prior) == len(legal_probs):
            legal_probs_sum = np.sum(legal_probs)
            if legal_probs_sum > 0:
                legal_probs = legal_probs / legal_probs_sum
            legal_probs = (
                (1.0 - HEURISTIC_PRIOR_WEIGHT) * legal_probs
                + HEURISTIC_PRIOR_WEIGHT * prior
            )

        act_probs = zip(legal_positions, legal_probs)
        return act_probs, value

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        self.policy_value_net.eval()
        with torch.no_grad():
            if self.use_gpu:
                state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
                log_act_probs, value = self.policy_value_net(state_batch)
                act_probs = np.exp(log_act_probs.data.cpu().numpy())
                return act_probs, value.data.cpu().numpy()
            else:
                state_batch = torch.FloatTensor(np.array(state_batch))
                log_act_probs, value = self.policy_value_net(state_batch)
                act_probs = np.exp(log_act_probs.data.numpy())
                return act_probs, value.data.numpy()

    def train_step(
        self,
        state_batch,
        mcts_probs,
        winner_batch,
        lr,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
    ):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
            mcts_probs = torch.FloatTensor(np.array(mcts_probs)).to(self.device)
            winner_batch = torch.FloatTensor(np.array(winner_batch)).to(self.device)
        else:
            state_batch = torch.FloatTensor(np.array(state_batch))
            mcts_probs = torch.FloatTensor(np.array(mcts_probs))
            winner_batch = torch.FloatTensor(np.array(winner_batch))

        self.policy_value_net.train()

        # zero the parameter gradients
        optimizer = self._optimizer_for_lr(lr)
        optimizer.zero_grad()

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the value head output is tanh, so it's in [-1, 1]
        # winner_batch is also in [-1, 1]
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss_weight * value_loss + policy_loss_weight * policy_loss
        
        # backward and optimize
        loss.backward()
        optimizer.step()
        
        # entropy for monitoring
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        self.last_train_components = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss_weight": float(policy_loss_weight),
            "value_loss_weight": float(value_loss_weight),
        }
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
