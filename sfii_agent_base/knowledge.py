import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs=6):
        super(ActorCritic, self).__init__()

        self.model_structure(num_inputs, num_outputs)
        self.initialize_weights_and_biases()

        self.train()

    def normalized_columns_initializer(self, weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
        return out

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)

    def model_structure(self, n_inp, n_out):
        self.conv1 = nn.Conv2d(n_inp, 32, kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=1)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(512, 120)

        self.critic_linear = nn.Linear(120, 1)
        self.actor_linear = nn.Linear(120, n_out)

    def initialize_weights_and_biases(self):
        self.apply(self.weights_init)
        self.actor_linear.weight.data = self.normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = self.normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, inputs):
        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = x.view(-1, 512)

        x = self.relu(self.fc1(x))

        return self.critic_linear(x), self.actor_linear(x)


class Knowledge():
    def __init__(self, input_frames, action_space):
        self.GAMMA = 0.9
        self.TAU = 1.0
        self.ENTROPY_COEF = 0.01
        self.VALUE_LOSS_COEF = 0.5

        self.model = ActorCritic(input_frames, num_outputs=action_space)
        self.optimizer = None

    def ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def initialize_optimizer(self, shared_agent):
        self.optimizer = optim.Adam(
            shared_agent.get_model().parameters(), lr=0.00001)

    def get_model(self):
        return self.model

    def load_model(self, model):
        self.model.load_state_dict(model.state_dict())

    def load_model_from_path(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def get_action(self, state):
        value, logit = self.model(torch.FloatTensor(state).unsqueeze(0))
        prob = F.softmax(logit, -1)
        action = prob.multinomial(num_samples=1)

        return action, value

    def train(self, experiences, game_finished, shared_agent):
        first_state = experiences[0, 0]
        n_experiences = len(experiences)

        value, logit = self.model(torch.FloatTensor(first_state).unsqueeze(0))
        prob = F.softmax(logit, -1)
        log_prob = F.log_softmax(logit, -1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)

        action = prob.multinomial(num_samples=1)
        log_prob = log_prob.gather(1, Variable(action))

        rewards = experiences[:, 2]
        entropies = list(
            entropy.view(-1).repeat(n_experiences).detach().numpy())
        log_probs = list(
            log_prob.view(-1).repeat(n_experiences).detach().numpy())

        values = []
        for i in range(n_experiences):
            values.append(value)

        # Check the final value
        final_state = experiences[0, -1]
        R = torch.zeros(1, 1)
        if not game_finished:
            value, _ = self.model(torch.FloatTensor(final_state).unsqueeze(0))
            R = value

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = self.GAMMA * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = (
                rewards[i] + self.GAMMA * values[i + 1].data - values[i].data)
            gae = gae * self.GAMMA * self.TAU + delta_t
            policy_loss = (
                policy_loss - (log_probs[i] * Variable(gae)) -
                (self.ENTROPY_COEF * entropies[i]))

        self.optimizer.zero_grad()
        loss_fn = (policy_loss + self.VALUE_LOSS_COEF * value_loss)
        loss_fn.backward()
        self.ensure_shared_grads(self.model, shared_agent.get_model())
        self.optimizer.step()
        torch.save(shared_agent.get_model().state_dict(), 'models/sf2_a3c.pth')
