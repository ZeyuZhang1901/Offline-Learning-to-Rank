# Offline Policy Learning (OPL) with DQN

## Introduction

A simple attempt of DQN alg. over OPL task.

## Algorithm

In the experiment, we use DQN as our training algorithm. ***The process to generate a ranking list under some query*** is modeled as an MDP, whose elements are defined as below:

- ***step***: each position in the ranking list, based on hyperparameter `END_POS`.
- ***state***: the **weighted** average feature of docs in previous and current positions, the closer to current position, the weight is heavier (*1/distance*). 
- ***action***: the feature of chosen doc at current position.
- ***candidates***: the doc set at current position under certain query. Once chosen as action, the document will be deleted.
- ***policy***: maximizing Q(**s,a**), which is implemented by neural network. 
- ***next state***: next state after chosen action by policy.
- ***reward***: matric dcg@k. In training session, we use click as reward signal, and subtract oracle propensity to debias; in evaluating session, true relevance label is used.

In DQN, Q-value refers to the expected rewards in some state and implements certain action. As a VisualizeResults_DoubleDQN, In our experiment, it means that after observing the documents at current position and before, the expected reward after placing certain document at the position. 

## Procedure

### Step1. Generate offline click dataset from cascade user interaction

Firstly, create ***offline click dataset*** by simulation: 

1. ***Pretrain Ranker*** with Online Methods, e.g. Online MDPRanker, and get logging policy.
   
2. ***Generate VisualizeResults_DoubleDQN list*** for each query with the logging policy, and ***simulate user clicks*** via our user model (Cascade assumption).
   
   - ![img](https://lh3.googleusercontent.com/yWX3vVuPzSEv-5YSEY58TbsSll1v6g0W3R9zoGQX-fVituJTqg2PBBrbGCHQJxK9A_mpeHNMJui1oFmQ_VP0myVuvaza2M46vsykCjaVTX1dJ-9M8gdW_MQ7h7MFRkBFMwFQaAwu-I703oW79ZcpWr8)
   - User satisfactory is modeled by position-dependent parameters $\lambda_i$ .
   
3. ***generate offline click dataset*** in the following format

   ```python
   click_dataset = {
       qid1: {
           "clicked_doces":clicked_doces,  # np.array, features of all candidates, vertically stacked ([num_doc, feature_dim])
           "click_labels":np.array(click_labels),  # np.array, binary click signal at each position in the session
           "propensities":obs_probs  # propensity from user model, e.g. DCM. 
       }
       qid2: {
           "clicked_doces":clicked_doces,
           "click_labels":np.array(click_labels),
           "propensities":obs_probs
       }
       ...
   }
   ```


### Step2. Offline training via DQN algorithm

1. ***Offline training settings***:

   - ***Model Structure***:

     ```python
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
     ```
     
     both applied to *policy network* and *target network*
     
   - ***Optimizer***: `optim.Adam` with `lr=1e-3`
   
   - ***Loss***: `nn.MSELoss()`
   
   - ***Replay Buffer***: 
   
     ```python
     Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'candidates', 'reward'))
     
     class ReplayMemory(object):
         '''Replay memory to store experiences'''
     
         def __init__(self, capacity) -> None:
             self.memory = deque([], maxlen=capacity)
     
         def push(self, *args):
             '''Save a transition in memory buffer'''
             self.memory.append(Transition(*args))
     
         def sample(self, batch_size):
             return random.sample(self.memory, batch_size)
     
         def __len__(self):
             return len(self.memory)
     ```
   

      - ***hyperparameters***

        ```python
        HYPERPARAMETERS = dict(
            STATE_DIM = 46,
            ACTION_DIM = 46,  
        
            # Offline RL training hyperparams
            BATCH_SIZE = 1024,
            GAMMA = 0.99,
            EPS_START = 0.9,
            EPS_END = 0.05,
            EPS_DECAY = 200000,
            TARGET_UPDATE = 100,
            OFFLINE_LR = 1e-3,
            OFFLINE_EPOCHS = 5000,
            ROLLOUT_TRAJ = 200,  # in each epoch, roll_out times
            N_EVAL = 10,  # sample number of evaluation
            EVAL_EPISODE = 10,  # evaluate every EVAL_EPISODE epochs
            MEMORY_SIZE = 700000,
        
            # Online training hyperparams
            ONLINE_LR = 1e-3,
            ONLINE_EPOCHS = 10,
            CLICK_MODEL = ['perfect', 'informational', 'navigational'],
            REWARD_METHOD = 'both', 
            END_POS = 10
        )
        ```
        
   
2. ***Train and evaluate for*** `OFFLINE_EPOCHS=5000` ***epochs. In each epoch***, 

   1. ***train batches***: sample `BATCH_SIZE=1024` tuples, each containing:

      ```python
   Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'candidates', 'reward'))
      ```
   
   3. ***evaluate***: sample `N_EVAL=10` trajectories, and compute *average ndcg* over the trajectories. 

3. ***plot and save model***: record performance via `SummaryWriter()`, and save model by `torch.save()`.

## Run Experiment

Directly run by `python main.py log_dir alg_type`, where `log_dir` is the logging path, str; and `alg_type` is the algorithm used for training, selected from `DQN, DoubleDQN`.

## Results

The logs (including train log and evaluation log) are in `log_dir/model_type` folder. Besides, visualization is also available by running `tensorboard --logdir=log_dir_alg_type`, and find VisualizeResults_DoubleDQNs at http://localhost:6006/.

```python
log_dir: str, logging path
model_type: str, ['perfect', 'informational', 'navigational']
alg_type: str, ['DQN', 'DoubleDQN']
```

|                   |                             loss                             |                           Q value                            |                           evaluate                           |                        evluate smooth                        |
| :---------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    **perfect**    | ![loss](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/perfect/Loss%20per%20Epoch.png) | ![Q_values](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/perfect/Q%20Value%20per%20Epoch.png) | ![evaluations](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/perfect/ndcg%20(Evaluation).png) | ![evaluations smooth](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/perfect/ndcg%20(Evaluation)%20smooth.png) |
| **informational** | ![loss](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/informational/Loss%20per%20Epoch.png) | ![Q_values](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/informational/Q%20Value%20per%20Epoch.png) | ![evaluations](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/informational/ndcg%20(Evaluation).png) | ![evaluations smooth](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/informational/ndcg%20(Evaluation)%20smooth.png) |
| **navigational**  | ![loss](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/navigational/Loss%20per%20Epoch.png) | ![Q_values](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/navigational/Q%20Value%20per%20Epoch.png) | ![evaluations](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/navigational/ndcg%20(Evaluation).png) | ![evaluations smooth](https://github.com/ZeyuZhang1901/offline_DQN_LTR/blob/action_redefine/VisualizeResults_DoubleDQN/navigational/ndcg%20(Evaluation)%20smooth.png) |

