from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.normal as normal
import random


class DQN(nn.Module):
    r'''Q-network implementation

        `Input`: state(shape=[batch_size, STATE_DIM])
                action(shape=[batch_size, ACTION_DIM])
        `Output`: Q value(shape=[batch_size, 1])'''

    def __init__(self, 
                state_dims, 
                action_dims,
                device) -> None:
        super(DQN, self).__init__()
        self.device = device
        self.ln1 = nn.Linear(state_dims + action_dims, 64, dtype=torch.float32)
        self.ln2 = nn.Linear(64, 32, dtype=torch.float32)
        self.output = nn.Linear(32, 1, dtype=torch.float32)
        self.net = nn.Sequential(
            self.ln1,
            nn.ELU(),
            self.ln2,
            nn.ELU(),
            self.output,
        )

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        return self.net(torch.cat((state, action), dim=1))

class Encoder(nn.Module):
    r'''encoder net in VAE
    
    `Input`: state(shape=[batch_size, STATE_DIM])
            action(shape=[batch_size, ACTION_DIM])
    `Output`: the expectation and standard deviaton of each state-action pair'''

    def __init__(self,
                state_dims,
                action_dims,
                device) -> None:
        super(Encoder, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(state_dims + action_dims, 750, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(750, 750, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(750, 2 * action_dims, dtype=torch.float32)
        )
    
    def forward(self,
                state, 
                action):
        state = state.to(self.device)
        action = action.to(self.device)
        return self.encoder(torch.cat((state, action), dim=1))

class Decoder(nn.Module):
    r'''decoder net in VAE
    
    `Input`: state(shape=[batch_size, STATE_DIM])
            latent_vec(shape=[batch_size, ACTION_DIM]): if not given, randomly selected from standard normal distribution
    `Output`: generated action based on state encoding'''

    def __init__(self,
                state_dims,
                latent_dims,
                device) -> None:
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.device = device
        self.decoder = nn.Sequential(
            nn.Linear(state_dims + latent_dims, 750, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(750, 750, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(750, latent_dims, dtype=torch.float32)
        )
    
    def forward(self,
                state, 
                latent_vec=None):
        if latent_vec == None:
            latent_vec = torch.randn((state.shape[0], self.latent_dims)).to(self.device).clamp(-0.5,0.5)
        state = state.to(self.device)
        latent_vec = latent_vec.to(self.device)
        return self.decoder(torch.cat((state, latent_vec), dim=1))

class VAE(object):
    r'''VAE is defined by two networks -- `encoder` and `decoder`.

        `encoder E(s,a)`: inputs a state-action pair and outputs the mean
        and standard deviation of a Gaussian distribution, with which we
        can construct latent variable `z` through reparameterization trick.
        
        `decoder D(s,z)`: inputs state and latent variable and output an action.
        
        While training, the loss is constructed by L2 loss, along with the KL 
        divergence between standard normal distribution and normal distribution
        induced by encoder. See `Eq.28~30` in paper `Off-Policy Deep Reinforcement
        Learning without Exploration` for details.'''

    def __init__(self,
                memory,
                hyperparameters) -> None:
        r'''`memory`: replay buffer\
        `hyperparameters`: hyperparameters of whole training procedure
        '''

        super(VAE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dims = hyperparameters['STATE_DIM']
        self.action_dims = hyperparameters['ACTION_DIM']
        self.batch_size = hyperparameters['BATCH_SIZE']
        self.memory = memory
        self.encoder = Encoder(hyperparameters['STATE_DIM'],\
            hyperparameters['ACTION_DIM'], self.device).to(self.device)
        self.decoder = Decoder(hyperparameters['STATE_DIM'], \
            hyperparameters['ACTION_DIM'], self.device).to(self.device)
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()))
        self.loss = nn.MSELoss()

    def sample(self, next_state_batch):
        r'''sample n actions from VAE
        
        `next_state_batch`: all next_states from the sample batch'''

        # decode and get simulated action vectors for each state
        return self.decoder(next_state_batch, latent_vec=None)

    def train_vae(self,
                state_batch,
                action_batch):
        r'''train VAE in one epoch
        
        `state_batch`: all states from the sample batch
        `action_batch`: all actions from the sample batch'''

        # encode and get mean and deviation vector
        mean_and_deviation = self.encoder(state_batch, action_batch)

        # sample latent vector for each mean-deviation pair
        normal_dist = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        samples = normal_dist.sample(torch.tensor([self.batch_size])).to(self.device)
        latent_vecs = (mean_and_deviation[:,:self.action_dims] \
            + samples * mean_and_deviation[:,self.action_dims:])

        # decode and get simulated action vectors for each state
        sim_action_batch = self.decoder(state_batch, latent_vecs)

        # loss calculation
        loss_reconstruct = self.loss(action_batch, sim_action_batch)
        square = mean_and_deviation**2
        loss_KL = -0.5 * (2 * self.action_dims + torch.log(square[:,self.action_dims:])\
            .sum(dim=1, keepdim=True) - torch.sum(square, dim=1, keepdim=True))
        loss = loss_reconstruct + 1 / 2 / self.action_dims * torch.sum(loss_KL)

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Perturbation(nn.Module):
    r'''perturb each simulated action
    
    `Input`: state(shape=[batch_size, STATE_DIM])
            action(shape=[batch_size, ACTION_DIM])
    `Output`: the perturbation of the generated action to diversify selected action, though constraint'''

    def __init__(self,
                state_dims,
                action_dims,
                perterb_range) -> None:
        super(Perturbation, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.perterb_range = perterb_range
        self.perturbation = nn.Sequential(
            nn.Linear(state_dims + action_dims, 400, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(400, 300, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(300, action_dims, dtype=torch.float32),
            nn.Tanh()  # serve as constraint
        )

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        return self.perturbation(torch.cat((state, action), dim=1))*self.perterb_range
