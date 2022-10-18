import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from Models import VAE, Perturbation, DQN
import ReplayBuffer as R

class Agent(object):   
    def __init__(self, 
                alg_name,
                hyperparameters, 
                memory) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hyperparameters = hyperparameters
        self.memory = memory
        if alg_name == 'BCQ':  # train BCQ agent
            self.vae_net = VAE(memory=memory,
                            hyperparameters=hyperparameters)
            self.perturbation = Perturbation(state_dims=hyperparameters['STATE_DIM'],
                            action_dims=hyperparameters['ACTION_DIM'],
                            perterb_range=self.hyperparameters['PERTURB_RANGE']).to(self.device)
            self.perturbation_target = Perturbation(state_dims=hyperparameters['STATE_DIM'],
                            action_dims=hyperparameters['ACTION_DIM'],
                            perterb_range=self.hyperparameters['PERTURB_RANGE']).to(self.device)
            self.perturbation_target.load_state_dict(self.perturbation.state_dict())
            self.perturbation_target.eval()
            self.policy_net = DQN(state_dims = hyperparameters['STATE_DIM'], 
                                action_dims = hyperparameters['ACTION_DIM'],
                                device = self.device).to(self.device)
            self.target_net1 = DQN(state_dims = hyperparameters['STATE_DIM'], 
                                action_dims = hyperparameters['ACTION_DIM'],
                                device = self.device).to(self.device)
            self.target_net2 = DQN(state_dims = hyperparameters['STATE_DIM'], 
                                action_dims = hyperparameters['ACTION_DIM'],
                                device = self.device).to(self.device)
            self.target_net1.load_state_dict(self.policy_net.state_dict())
            self.target_net1.eval()
            self.target_net2.load_state_dict(self.policy_net.state_dict())
            self.target_net2.eval()
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(),
                lr=self.hyperparameters['LR'])
            self.perturbation_optimizer = optim.Adam(self.perturbation.parameters(),
                lr=self.hyperparameters['LR'])
            self.policy_loss = nn.MSELoss()
        elif alg_name == 'DQN' or alg_name == 'DoubleDQN':  # train DQN / DoubleDQN
            self.policy_net = DQN(state_dims = hyperparameters['STATE_DIM'], 
                            action_dims = hyperparameters['ACTION_DIM'],
                            device = self.device).to(self.device)
            self.target_net = DQN(state_dims = hyperparameters['STATE_DIM'], 
                            action_dims = hyperparameters['ACTION_DIM'],
                            device = self.device).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(),
                lr=self.hyperparameters['LR'])
            self.loss = nn.MSELoss()

        self.eval_count = 1

    def selectAction(self,
                    state,  # current state (weighted average of previous selected docs)
                    candidates):  # current docs set (shape: [num_candidates, action_dim])
        '''select action from all candidates, based on epsilon-greedy'''

        scores = self.policy_net(state.expand(candidates.shape[0], -1), candidates)
        return candidates[torch.max(scores, 0)[1]].view(1, self.hyperparameters['ACTION_DIM'])

    def getTargetMaxValue(self,
                        state,
                        candidates):
        '''get max Q value given state and candidates'''

        scores = self.target_net(state.expand(candidates.shape[0], -1), candidates)
        return torch.max(scores, 0)[0].item()

    def getCandidateFeatures(self,
                            dataset,  # train set or test set
                            qid):  # query id
        '''get features of all candidates docs under `qid`, return tensor in shape `[num_doc, ACTION_DIM]`'''

        return torch.tensor(dataset.get_all_features_by_query(qid), dtype=torch.float32)
                
    def DQNTrainOneStep(self,
                        dataset):
        '''train one batch via DQN alg'''

        trainsitions = self.memory.sample(self.hyperparameters['BATCH_SIZE'])
        batch = R.Transition(*zip(*trainsitions))
        
        # gather batch 
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # get Q(s,a) from policy network
        state_action_values = self.policy_net(state_batch, action_batch)
        # get Q(s',a') from target network, maximizing target Q value
        next_state_values = torch.zeros(self.hyperparameters['BATCH_SIZE'],1,\
            dtype=torch.float32, device=self.device).detach()
        for i in range(self.hyperparameters['BATCH_SIZE']):
            candidates = self.getCandidateFeatures(dataset, batch.qid[i])[batch.chosen[i]]
            next_state_values[i,0] = self.getTargetMaxValue(next_state_batch[i], candidates)
        # expected Q values from bootstrapping
        expected_state_action_values = next_state_values * self.hyperparameters['GAMMA']\
                                    + reward_batch  

        # loss and optimization
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network soft update
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
           target_param.data.copy_(self.hyperparameters['TAU'] * param.data + \
            (1 - self.hyperparameters['TAU']) * target_param.data)

        return state_action_values.mean().item(), \
            expected_state_action_values.mean().item(), \
            loss.item()

    def doubleDQNTrainOneStep(self,
                            dataset):
        '''train one batch via double DQN alg'''

        trainsitions = self.memory.sample(self.hyperparameters['BATCH_SIZE'])
        batch = R.Transition(*zip(*trainsitions))
        
        # gather batch  
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # double DQN: actions selected in the target Q network comes from policy network
        # get Q(s,a) from policy network
        state_action_values = self.policy_net(state_batch, action_batch)
        # get action from policy network in case overestimation
        next_state_actions = torch.zeros(self.hyperparameters['BATCH_SIZE'],\
            self.hyperparameters['ACTION_DIM'], dtype=torch.float32)
        for i in range(self.hyperparameters['BATCH_SIZE']):
            candidates = self.getCandidateFeatures(dataset, batch.qid[i])[batch.chosen[i]]
            next_state_actions[i] = self.selectAction(state = next_state_batch[i], 
                candidates = candidates if candidates.shape[0] != 0 \
                else torch.zeros(1,self.hyperparameters['ACTION_DIM'], dtype=torch.float32)) # no action to choose
        # Q value from target network
        next_state_values = self.target_net(next_state_batch, next_state_actions)
        # expected Q values from bootstrapping
        expected_state_action_values = reward_batch + \
            next_state_values * self.hyperparameters['GAMMA']

        # loss and optimization
        loss = self.loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network soft update
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
           target_param.data.copy_(self.hyperparameters['TAU'] * param.data + \
            (1 - self.hyperparameters['TAU']) * target_param.data)

        return state_action_values.mean().item(), \
            expected_state_action_values.mean().item(), \
            loss.item()

    def BCQTrainOneStep(self):
        '''train one batch via double BCQ alg'''

        trainsitions = self.memory.sample(self.hyperparameters['BATCH_SIZE'])
        batch = R.Transition(*zip(*trainsitions))

        # gather batch  
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # train VAE to get action generator
        self.vae_net.train_vae(state_batch, action_batch)

        # sample new actions under next states and add perturbation
        next_state_batch = torch.repeat_interleave(next_state_batch, self.hyperparameters['ACTION_SAMPLE'], 0)
        sim_action_batch = self.vae_net.sample(next_state_batch)
        action_with_perturb = sim_action_batch + \
            self.perturbation(next_state_batch, sim_action_batch)
        
        # policy net optimization
        # set target and calculate loss
        target_Q1= self.target_net1(next_state_batch, action_with_perturb)
        target_Q2 = self.target_net2(next_state_batch, action_with_perturb)
        target_Q = self.hyperparameters['TARGET_LAMBDA'] * torch.min(target_Q1, target_Q2)\
            + (1 - self.hyperparameters['TARGET_LAMBDA']) * torch.max(target_Q1, target_Q2)
        target_Q = target_Q.reshape(self.hyperparameters['BATCH_SIZE'],-1).max(1)[0].reshape(-1,1)
        target_value = reward_batch + self.hyperparameters['GAMMA'] * target_Q

        policyloss = self.policy_loss(self.policy_net(state_batch, action_batch), target_value)
        self.policy_optimizer.zero_grad()
        policyloss.backward()
        self.policy_optimizer.step()

        # perturbation optimization
        # set target and calculate loss
        action_with_perturb_target = action_batch + \
            self.perturbation(state_batch, action_batch)
        perturbation_target = -torch.mean(self.policy_net(state_batch, action_with_perturb_target))

        self.perturbation_optimizer.zero_grad()
        perturbation_target.backward()
        self.policy_optimizer.step()

        # target network soft update
        for param, target1_param in zip(self.policy_net.parameters(), self.target_net1.parameters()):
           target1_param.data.copy_(self.hyperparameters['TAU'] * param.data + \
            (1 - self.hyperparameters['TAU']) * target1_param.data)
        for param, target2_param in zip(self.policy_net.parameters(), self.target_net2.parameters()):
           target2_param.data.copy_(self.hyperparameters['TAU'] * param.data + \
            (1 - self.hyperparameters['TAU']) * target2_param.data)
        for param, perturb_param in zip(self.perturbation.parameters(), self.perturbation_target.parameters()):
           perturb_param.data.copy_(self.hyperparameters['TAU'] * param.data + \
            (1 - self.hyperparameters['TAU']) * perturb_param.data)
        
        return self.policy_net(state_batch, action_batch).mean().item(),\
            target_value.mean().item(),\
            policyloss.item()

    def evaluate(self,
                queryset,
                dataset,
                end_position=10):
        '''evaluate policy by ndcg@k, where k is the endpoint of the list'''

        total_ndcg = 0
        nan_count = 0
        print(f"Evaluate {self.eval_count}: ")

        qids = np.random.choice(queryset, self.hyperparameters['N_EVAL'], replace=False)
        traj_count = 1
        for qid in qids:
            candidates = self.getCandidateFeatures(dataset, qid)  # all candidates
            query_steps = np.min([candidates.shape[0], end_position]) \
                if end_position != None else candidates.shape[0]
            relvance = np.zeros(query_steps)  # record relvances

            next_state = None
            for position in range(0,query_steps):
                # state
                state = torch.zeros(1,self.hyperparameters['STATE_DIM'], dtype=torch.float32)\
                    if position == 0 else next_state
                # action 
                action = self.selectAction(state=state,  # in case no candidates
                    candidates=candidates if candidates.shape[0] != 0\
                    else torch.zeros(1,self.hyperparameters['ACTION_DIM'], dtype=torch.float32))
                # reward
                docid = dataset.get_docid_by_query_and_feature(qid, action.view(-1).numpy())
                if docid == None:
                    relvance[position] = 0
                else:
                    relvance[position] = dataset.get_relevance_label_by_query_and_docid(qid,docid)
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
                # print(f'traj{traj_count+1}(qid = {qid})\trewards = {round(dcg,6)}\tmax_rewards = {round(max_dcg, 6)}\tnormal_rewards = {round(ndcg, 6)}')
            else:
                nan_count += 1
                # print(f"traj{traj_count+1}(qid = {qid})\tNo relevant documents.")
            traj_count += 1
        
        avg_ndcg = total_ndcg/(self.hyperparameters['N_EVAL'] - nan_count)
        print(f"average_ndcg = {round(avg_ndcg, 6)}")
        self.eval_count += 1
        return avg_ndcg

    def modelSave(self, path):
        torch.save(self.policy_net, path)

    def writeEvalLog(self, filepath, model, lines, previous_eval):
        with open(filepath + '/' + str(model)+'/evaluation_log.txt', 'a') as f:
            f.write(f"Evaluate {self.eval_count + previous_eval}: \n")
            f.writelines(lines)

    def writeTrainLog(self, filepath, model, lines):
        with open(filepath + '/' + str(model)+'/train_log.txt', 'a') as f:
            f.writelines(lines)
                