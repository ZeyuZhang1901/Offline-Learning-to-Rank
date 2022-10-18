import random
import torch
import numpy as np
import argparse

from PretrainRanker.dataset.Preprocess import PreprocessDataset
from Models import DQN
from TrainTest import Agent

# Seed fixed
seed = 46
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
# parser.add_argument('--path', type=str, required=True)
parser.add_argument('--alg', type=str, required=True)
args = parser.parse_args()

EVAL_HYPERPARAMETERS = dict(
    STATE_DIM = 46,
    ACTION_DIM = 46,  

    CLICK_MODEL = ['perfect', 'informational', 'navigational'],
    END_POS = 10
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def selectAction(policy,
                state,  # current state (weighted average of previous selected docs)
                candidates):  # current docs set (shape: [num_candidates, action_dim])
    '''select action from all candidates, based on epsilon-greedy'''

    if policy == None:  # randomly select an action from candidates
        return candidates[random.choice(range(candidates.shape[0]))]\
            .view(1,EVAL_HYPERPARAMETERS['ACTION_DIM'])
    else:
        scores = policy(state.expand(candidates.shape[0], -1), candidates)
        return candidates[torch.max(scores, 0)[1]].view(1, EVAL_HYPERPARAMETERS['ACTION_DIM'])

def getCandidateFeatures(dataset,  # train set or test set
                        qid):  # query id
    '''get features of all candidates docs under `qid`, return tensor in shape `[num_doc, ACTION_DIM]`'''

    return torch.tensor(dataset.get_all_features_by_query(qid), dtype=torch.float32)

def policyTest(policy, 
            queryset, 
            dataset, 
            end_position):
    '''evaluate policy on all queries in the test set, matric is ndcg@k'''

    total_ndcg = 0
    nan_count = 0
    lines = []

    traj_count = 1
    for qid in queryset:
        candidates = getCandidateFeatures(dataset, qid)  # all candidates
        query_steps = np.min([candidates.shape[0], end_position]) \
            if end_position != None else candidates.shape[0]
        relvance = np.zeros(query_steps)  # record relvances

        next_state = None
        for position in range(0,query_steps):
            # state
            state = torch.zeros(1,EVAL_HYPERPARAMETERS['STATE_DIM'], dtype=torch.float32)\
                if position == 0 else next_state
            # action 
            action = selectAction(policy, state, candidates)
            # reward
            relvance[position] = dataset.get_relevance_label_by_query_and_docid(qid,\
                dataset.get_docid_by_query_and_feature(qid, action.view(-1).numpy()))
            # next state
            next_state = action + position*state/(position+1)
            # delete chosen doc in candidates
            indices = ((candidates == action[:, None]).sum(axis = 2)\
                != candidates.shape[1]).all(axis = 0)
            candidates = candidates[indices]

        # evaluate ndcg matric
        dcg = np.sum((np.power(2,relvance) - 1)/np.log2(np.arange(2,query_steps+2)))
        all_relevance = np.array(dataset.get_all_relevance_label_by_query(qid))
        sort_relevance = np.flip(np.sort(all_relevance))
        max_dcg = np.sum((np.power(2,sort_relevance[:query_steps]) - 1)/np.log2(np.arange(2,query_steps+2)))
        if max_dcg !=0:
            ndcg = dcg/max_dcg
            total_ndcg += ndcg
            line = f'traj{traj_count}(qid = {qid})\trewards = {round(dcg,6)}\tmax_rewards = {round(max_dcg, 6)}\tnormal_rewards = {round(ndcg, 6)}'
            # print(line)
            lines.append(line+'\n')
        else:
            nan_count += 1
            line = f"traj{traj_count}(qid = {qid})\tNo relevant documents."
            # print(line)
            lines.append(line+'\n')
        traj_count += 1
    
    avg_ndcg = total_ndcg/(len(queryset) - nan_count)
    line = f"average_ndcg = {round(avg_ndcg, 6)}"
    # print(line)
    lines.append(line+'\n')
    return avg_ndcg, lines



if __name__ == "__main__":
    dataset_fold = "./PretrainRanker/dataset/MQ2007"
    test_path = "{}/Fold1/test.txt".format(dataset_fold)
    test_set = PreprocessDataset(test_path,
                                EVAL_HYPERPARAMETERS['STATE_DIM'],
                                query_level_norm=False)
    queryset = test_set.get_all_querys()
    
    for _ in range(5):
        for model in EVAL_HYPERPARAMETERS['CLICK_MODEL']:
            # path = args.path + f'_{args.alg}'
            # policy = torch.load(f'{path}/{model}/myModel_{model}.pth')
            # print(list(policy.parameters()))
            # test_log = f'{path}/{model}/test_log.txt'
            policy = DQN(state_dims = EVAL_HYPERPARAMETERS['STATE_DIM'], 
                action_dims = EVAL_HYPERPARAMETERS['ACTION_DIM'],
                device = device).to(device)
            avg_ndcg, lines =  policyTest(policy=policy,
                                        queryset=queryset, 
                                        dataset=test_set, 
                                        end_position=EVAL_HYPERPARAMETERS['END_POS'])
            with open('validation.txt', 'a') as f:
                f.write(f'average ndcg on {model} model: {avg_ndcg}\n')
        
        with open('validation.txt', 'a') as f:
            f.write(f'\n')
    
    with open('validation.txt', 'a') as f:
        f.write(f'\n-------------------------------------------------------------------\n')
    
    for _ in range(5):
        for model in EVAL_HYPERPARAMETERS['CLICK_MODEL']:
            # path = args.path + f'_{args.alg}'
            # policy = torch.load(f'{path}/{model}/myModel_{model}.pth')
            # policy = DQN(state_dims = EVAL_HYPERPARAMETERS['STATE_DIM'], 
            #                 action_dims = EVAL_HYPERPARAMETERS['ACTION_DIM'],
            #                 device = device).to(device)
            # print(list(policy.parameters()))
            # test_log = f'{path}/{model}/test_log.txt'
            avg_ndcg, lines =  policyTest(policy=None,
                                        queryset=queryset, 
                                        dataset=test_set, 
                                        end_position=EVAL_HYPERPARAMETERS['END_POS'])
            with open('validation.txt', 'a') as f:
                f.write(f'average ndcg on {model} model: {avg_ndcg}\n')

        with open('validation.txt', 'a') as f:
            f.write(f'\n')


            