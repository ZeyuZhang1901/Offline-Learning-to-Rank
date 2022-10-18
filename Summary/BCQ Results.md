## BCQ Results

We implement ablation exps with respect to the following factors:

1. *click models*: we implement three different sets of click probability, `perfect [0, 0.5, 1]`, `informational [0.4, 0.7, 0.9]`, `navigational [0.05, 0.5, 0.95]`
2. *perturbation range*: perturbation is the key in BCQ, which constraints the searching and updating area. Intuitively, larger perturbation range leads to larger variance (loss) and greater probability; and a smaller one makes an opposite effect.

### 1. Click Models

To figure out the effect of click models, we fix perturbation as a single `tanh()` function, and then compared the loss, Q value and evaluation (ndcg@10) of different click models. Here are the results:

- *Loss per epoch*

![Loss per Epoch](/home/zeyuzhang/Pictures/Screenshot_20221015_122356.png)

- *Q value per epoch*

![Q value](/home/zeyuzhang/Pictures/Screenshot_20221015_122550.png)

- *ndcg@10 (evlauted after every ten epochs)*

![Evluation ndcg](/home/zeyuzhang/Pictures/Screenshot_20221015_122647.png)

As can be seen in fig1, loss in all three models increases rapidly as the number of epochs increases, where the loss under *informational* model is the most noticeable, much higher than the other two models. Correspondingly, *informational* model has the largest Q value. However, what surprises me most is that the algorithms get the best performance under *informational*  model, which is unusual as *perfect* model can unbiasedly represent the click (click probability is propotional to relevance label). Further analysis can be seen in **Analysis** section (section three).

### 2. Perturbation Range

Similar to exps in **Click Models** part, we fix the click model and change perturbation range by *multiplying a factor to scale the range of function `tanh()`*. We only show the loss and evaluation under each model, and here are the results:

- perfect model

  ![Loss per epoch](/home/zeyuzhang/Pictures/Screenshot_20221015_125142.png)

  ![evaluation per 10 epochs](/home/zeyuzhang/Pictures/Screenshot_20221015_125223.png)

- informational model

  ![Loss per epoch](/home/zeyuzhang/Pictures/Screenshot_20221015_125428.png)

  ![Evaluation per 10 epochs](/home/zeyuzhang/Pictures/Screenshot_20221015_125508.png)

- navigational model

![Loss per epoch](/home/zeyuzhang/Pictures/Screenshot_20221015_125711.png)

![evaluation per 10 epochs](/home/zeyuzhang/Pictures/Screenshot_20221015_125842.png)

Loss under all models doesn't converge when `perturb=2`, and will be reasonably bounded to prevent overestimate if perturbation factor is small (e.g. `perturb=0.5`).  Though the loss can be controlled, small perturbation factor may lead to other problems, like *slower convergence rate*, *suboptimal solution*. 

### 3. Analysis

Based on the two exps in the past two sections, I think that perturbation works as a ***"trade-off"*** parameter. 

- Larger perturbation means that more actions can be reached, leading to greater probability of finding optimal solution(s). However, this also means that some unrelated actions will be taken into consideration, which will cause severe *overestimation* (mismatch in the distribution of data induced by the policy and the distribution of data contained in the batch)
- Smaller perturbation add a constrain on the similarity between selected actions and actions in the batch, which will remarkably reduce the overestimation problem. However, Some other problems occurs, such as *slower convergence rate*, *suboptimal solution* , especially under click models with bias (e.g. informational and navigational). 

### Appendix

- ndcg@10, perturb=0.5

  ![ndcg@10, perturb=0.5](/home/zeyuzhang/Pictures/Screenshot_20221015_132259.png)

- ndcg@10, perturb=2

  ![ndcg@10, perturb=2](/home/zeyuzhang/Pictures/Screenshot_20221015_132520.png)