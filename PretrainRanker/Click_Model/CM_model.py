import sys
sys.path.append('./')

import numpy as np
from .AbstractClickModel import AbstractClickModel


class CM(AbstractClickModel):
    def __init__(self, pc=None, ps=None, alpha=1, beta=1):
        self.name = 'CM'
        self.parameter_dict = {}
        self.stat_dict = {}
        self.alpha = alpha
        self.beta = beta
        self.pc = pc
        self.ps = ps


    def set_probs(self, pc, ps):
        self.pc = pc
        self.ps = ps

    def simulate(self, query, result_list, dataset):
        clicked_doc = []
        click_label = np.zeros(len(result_list))
        satisfied = False
        for i in range(0, len(result_list)):  # for each doc in the result list
            click_prob = np.random.rand()  # generate a click_prob uniformly from (0,1]
            docid = result_list[i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)

            if click_prob <= self.pc[relevance]:
                click_label[i] = 1
                clicked_doc.append(result_list[i])
                satisfied = True
                break

        return clicked_doc, click_label, satisfied

    def simulate_cascade(self, query, result_list, dataset, cascade_type='DCM', k=10):
        length = len(result_list) if (k == None or k> len(result_list)) else k

        clicked_doc = None
        click_start = False
        click_label = np.zeros(length)
        obs_probs = np.zeros(length)  
        obs_probs[0] = 1  # in Cascade model, the first doc must be observed
        satisfied = False

        if cascade_type == 'DCM': 
            r'''
                user will continue searching the list until she clicks or the list is over
                a position dependent chance that the user not satisfied after each click, \lambda_j
                P(E_{j+1}=1 | E_j=1, C_j=0) = 1
                P(E_{j+1}=1 | C_j=1) = \lambda_j = \beta(\frac{1}{j})^\eta
            --> P_{DCM}(E_j=1 | c_{<j}) = \prod_{i<j}(1-c_i(1-\lambda_i))
            '''
            beta = 1  # beta =0.6 or 1
            eta = 0.5  # eta = 0.5 or 1 or 2   
            lambdas = beta * np.power(1/np.arange(1,len(result_list)+1),eta)

            for i in range(length): 
                # click through?
                if satisfied == False:
                    docid = result_list[i]
                    relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
                    if np.random.rand() <= self.pc[relevance]:
                        click_label[i] = 1
                        if click_start == False:
                            clicked_doc = dataset.get_features_by_query_and_docid(query, docid).reshape(1,-1)
                            click_start = True
                        else:
                            np.append(clicked_doc,\
                                dataset.get_features_by_query_and_docid(query, docid).reshape((1,-1)))
                        # See more results?
                        if i<length-1:
                            obs_probs[i+1] = np.prod(1 - click_label[:i+1] * (1-lambdas[:i+1]))
                            if np.random.rand() > obs_probs[i+1]:
                                satisfied = True
                    else:
                        if i<length-1:
                            obs_probs[i+1] = obs_probs[i]
                else:
                    if i<length-1:
                        obs_probs[i+1:] = obs_probs[i]
                        break

        elif cascade_type == 'DBN':
            r'''
                consider user satisfaction S_i, session might be abandoned
                satisfied user abandon the session: P(E_{i+1}=1 | S_i=1) = 0
                unsatisfied user abandon the session with constant prob: \gamma
                the user might not be satisfied even after a click: P(S_i=1 | C_i=1)=s_{x_i}
            --> P_{DBN}(E_j=1 | c_{<j}) = \prod_{i<j}\gamma\cdot(1-c_i\cdot s_{x_i})
            '''
            gamma = 0.1  # 10% chance abandoning the session even unsatisfied

            obs_probs = np.zeros(len(result_list)) 
            obs_probs[0] = 1  # in Cascade model, the first doc must be observed
            for i in range(len(result_list)):
                click_prob = np.random.rand()
                docid = result_list[i]
                relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)

                # judge whether to click
                if satisfied == False:  # not abandon
                    if click_prob <= self.pc[relevance]:
                        click_label[i] = 1
                        clicked_doc = dataset.get_features_by_query_and_docid(query, docid) \
                            if clicked_doc == None \
                            else np.append(clicked_doc,\
                                dataset.get_features_by_query_and_docid(query, docid).reshape((1,-1)))
                        # check whether satisfied
                        abandon_prob = np.random.rand()
                        if abandon_prob <= self.ps[relevance]:  # satisfied
                            satisfied = True
                        if i< len(result_list)-1:  # in case out of bounds
                            obs_probs[i+1] = obs_probs[i] * (1-gamma) * (1-click_label[i] * self.ps[relevance])
                        continue
                if i< len(result_list)-1:
                    obs_probs[i+1] = obs_probs[i] * (1-gamma)

        if click_start == False:  # no click
            clicked_doc = -1 * np.ones_like(dataset.get_features_by_query_and_docid(query, 0)).reshape(1,-1)
        return clicked_doc, click_label, obs_probs, click_start

    def train(self, click_log):
        self._get_train_stat(click_log)

        print("{} training.......".format(self.name))
        for qid in self.stat_dict.keys():
            self.parameter_dict[qid] = {}
            for docID in self.stat_dict[qid].keys():
                a = (self.stat_dict[qid][docID][1] + self.alpha) / (self.stat_dict[qid][docID][0] + self.alpha + self.beta)
                self.parameter_dict[qid][docID] = a

    def _get_train_stat(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]
        for line in range(dataset_size):

            qid = click_log[line][0]
            docIds = click_log[line][1:11]
            clicks = click_log[line][11:21]

            if qid not in self.stat_dict.keys():
                self.stat_dict[qid] = {}

            doc_stat = self.stat_dict[qid]

            if np.where(clicks == '1')[0].size == 0:
                continue

            lastClickRank = np.where(clicks == '1')[0][-1] + 1

            for rank in range(lastClickRank):
                docID = docIds[rank]
                if docID not in doc_stat.keys():
                    doc_stat[docID] = (0, 0)
                exam = doc_stat[docID][0] + 1
                c = doc_stat[docID][1]
                if clicks[rank] == '1':
                    c += 1

                doc_stat[docID] = (exam, c)
            # if line % 10000 == 0:
            #     print("process %d/%d of dataset" % (line, dataset_size))

    def get_click_probs(self, session):
        qid = session[0]
        docIds = session[1:11]
        a_probs = np.zeros(10)
        exam_probs = np.zeros(10)
        exam_probs[0] = 1
        for i in range(1, 10):
            if docIds[i - 1] not in self.parameter_dict[qid].keys():
                ar = self.alpha / (self.alpha + self.beta)
            else:
                ar = self.parameter_dict[qid][docIds[i - 1]]
            exam_probs[i] = exam_probs[i - 1] * (1 - ar)

        for i in range(10):
            if docIds[i] not in self.parameter_dict[qid].keys():
                a = self.alpha / (self.alpha + self.beta)
            else:
                a = self.parameter_dict[qid][docIds[i]]
            a_probs[i] = a

        return np.multiply(exam_probs, a_probs)

    def get_real_click_probs(self, session, dataset):
        qid = session[0]
        docIds = session[1:11]
        exam_probs = np.zeros(10)
        exam_probs[0] = 1
        a_probs = np.zeros(10)

        for i in range(1, 10):
            relevance = dataset.get_relevance_label_by_query_and_docid(qid, int(docIds[i -1]))
            ar = self.pc[relevance]
            exam_probs[i] = exam_probs[i - 1] * (1 - ar)

        for i in range(10):
            relevance = dataset.get_relevance_label_by_query_and_docid(qid, int(docIds[i]))
            a = self.pc[relevance]
            a_probs[i] = a
        return np.multiply(exam_probs, a_probs)

    def get_perplexity(self, test_click_log):
        print(self.name, "computing perplexity")
        perplexity = np.zeros(10)
        size = test_click_log.shape[0]
        for i in range(size):
            session = test_click_log[i][:11]
            click_label = test_click_log[i][11:]
            click_probs = self.get_click_probs(session)
            for rank, click_prob in enumerate(click_probs):
                if click_label[rank] == '1':
                    p = click_prob
                else:
                    p = 1 - click_prob

                with np.errstate(invalid='raise'):
                    try:
                        p = 0.001 if p < 0.001 else p
                        perplexity[rank] += np.log2(p)
                    except:
                        print("error!, p=", p)
                        print(session, rank + 1)
                        perplexity[rank] += 0

        perplexity = [2 ** (-x / size) for x in perplexity]
        return perplexity

    def get_MSE(self, test_click_log, dataset, simulator):
        print(self.name, "computing MSE")
        MSE = np.zeros(10)
        size = test_click_log.shape[0]
        for i in range(size):
            session = test_click_log[i]
            click_probs = self.get_click_probs(session)
            real_click_probs = simulator.get_real_click_probs(session, dataset)
            MSE += np.square(click_probs - real_click_probs)

        return MSE/size
