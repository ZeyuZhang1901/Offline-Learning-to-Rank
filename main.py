import os
import sys
import argparse
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

from ReplayBuffer import ReplayMemory
from PretrainRanker.dataset.Preprocess import PreprocessDataset
from PretrainRanker.Ranker.MDPRankerV2 import MDPRankerV2
from TrainTest import Agent
from OfflineData import offlineDataCollect

# Seed fixed
seed = 46
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# running arguments setting
parser = argparse.ArgumentParser()
parser.add_argument('--log', default=False, action="store_true")
parser.add_argument('--previous-epoch', type=int, required=True)
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--alg', type=str, required=True)
args = parser.parse_args()
previous_epoch = args.previous_epoch

# Hyperparameters
HYPERPARAMETERS = dict(
    # State and action dim
    # STATE_DIM = 46,  # MQ2007/MQ2008
    # ACTION_DIM = 46,  
    STATE_DIM=136,  # MSLR_WEB10K
    ACTION_DIM=136,

    # Offline RL training hyperparams
    BATCH_SIZE = 256,
    GAMMA = 0.9,
    TARGET_UPDATE = 100,
    LR = 1e-3,
    EPOCHS = 5000,
    # EPOCHS = 10000,
    TAU = 0.005,  # soft target update

    # Hyperparameters related to sample
    # MQ2007/MQ2008 (300 queries)
    NUM_SAMPLE = 200,  # num of lists under each query 
    N_EVAL = 30,  # num of evaluation samples
    EVAL_EPISODE = 10,  # evaluate interval
    MEMORY_SIZE = 1000000,
    # MSLR_WEB10K/30K (10000 queries/30000 queries)
    # NUM_SAMPLE = 30,  # num of lists under each query 
    # N_EVAL = 30,  # num of evaluation samples
    # EVAL_EPISODE = 10,  # evaluate interval
    # MEMORY_SIZE = 2000000,
    
    # BCQ hyperparams
    TARGET_LAMBDA = 0.75,  # soft target trade-off parameter
    # PERTURB_RANGE = 1,  # perturb range of perturbation model
    # PERTURB_RANGE = 0.5,
    PERTURB_RANGE = 0.2,
    ACTION_SAMPLE = 10,  # num of action VAE action sample

    # Click model
    CLICK_MODEL = ['perfect', 'informational', 'navigational'],
    END_POS = 10  # ndcg@10
)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(model,  # click model type
        path,  # whole path of the exp
        alg,  # training alg
        previous_epoch, # int, previous training epochs
        memory,  # replay buffer
        train_set,  # train set
        test_set,  # test_set
        test_queryset,  # test_queryset
        log=False,  # bool, whether logging and writing
        save = False,  # bool, whether saving
        load=False):  # bool, whether load previous model

    # logger
    if log:
        writer = SummaryWriter(path+'/'+model+f"(perturb={HYPERPARAMETERS['PERTURB_RANGE']})")\
            if alg == 'BCQ' else SummaryWriter(path+'/'+model)
    
    agent = Agent(alg_name=alg,
                hyperparameters=HYPERPARAMETERS,
                memory=memory)

    # training start!
    print("\n**********Offline training start!**********")
    eval_count = 0
    for epoch in range(1,HYPERPARAMETERS['EPOCHS']+1):
        # train
        if alg == 'DQN':
            state_action_value, expected_state_action_value, loss = agent.DQNTrainOneStep(train_set)
        elif alg == 'DoubleDQN':
            state_action_value, expected_state_action_value, loss = agent.doubleDQNTrainOneStep(train_set)
        elif alg == 'BCQ':
            state_action_value, expected_state_action_value, loss = agent.BCQTrainOneStep()

        # writer and logger for train
        print(f"Epoch {epoch+previous_epoch}:\tloss = {round(loss, 6)}\tpolicy_Q_value = {round(state_action_value, 6)}\ttarget_Q_value = {round(expected_state_action_value, 6)}")
        if log:
            writer.add_scalars('Q Value per Epoch',
                    {'policy Q value': state_action_value,
                    'target Q value': expected_state_action_value,
                    'difference': state_action_value-expected_state_action_value},
                    previous_epoch + epoch+1)
            writer.add_scalar('Loss per Epoch',loss, previous_epoch+epoch+1)

        # evaluate
        if epoch % HYPERPARAMETERS['EVAL_EPISODE'] == 0:
            with torch.no_grad():
                ndcg = agent.evaluate(queryset=test_queryset,
                                        dataset=test_set,
                                        end_position=HYPERPARAMETERS['END_POS'])
                if log:
                    writer.add_scalar('ndcg (Evaluation)',ndcg, \
                        int(previous_epoch/HYPERPARAMETERS['EVAL_EPISODE']) + eval_count)
                    eval_count += 1

        # # renew target network parameters
        # if epoch % HYPERPARAMETERS['TARGET_UPDATE'] == 0:
        #     agent.target_net.load_state_dict(agent.policy_net.state_dict())
    if log:
        writer.close()

    print("**********Offline training finish!**********")
    

if __name__ == "__main__":
    # datasets and logging ranker

    # dataset_fold = "./PretrainRanker/dataset/MQ2007"
    dataset_fold = "./PretrainRanker/dataset/MQ2008"
    # dataset_fold = "./PretrainRanker/dataset/MSLR_WEB10K"
    training_path = "{}/Fold1/train.txt".format(dataset_fold)
    test_path = "{}/Fold1/test.txt".format(dataset_fold)

    # offline data preparation
    train_set = PreprocessDataset(training_path,
                                HYPERPARAMETERS['STATE_DIM'],
                                query_level_norm=False)
    train_queryset = train_set.get_all_querys()
    test_set = PreprocessDataset(test_path,
                                HYPERPARAMETERS['STATE_DIM'],
                                query_level_norm=False)
    test_queryset = test_set.get_all_querys()
    ranker = MDPRankerV2(256,
                        HYPERPARAMETERS['STATE_DIM'],
                        Learningrate=1e-3,
                        loss_type='pairwise')
    
    # for different click model
    for model in HYPERPARAMETERS['CLICK_MODEL']:
        replay_memory = ReplayMemory(HYPERPARAMETERS['MEMORY_SIZE'])
        offlineDataCollect(click_model = model,
                        hyperparameter = HYPERPARAMETERS,
                        train_set=train_set,
                        train_queryset=train_queryset,
                        ranker=ranker,
                        memory = replay_memory,
                        end_position = HYPERPARAMETERS['END_POS'])
        main(model = model,
            path = args.path+f'_{args.alg}',
            alg = args.alg,
            previous_epoch = previous_epoch,
            memory=replay_memory,
            train_set=train_set,
            test_set=test_set,
            test_queryset=test_queryset,
            log= args.log)