# stochastic_bayesflow
This repository supports the paper "Solving Stochastic Inverse Problem with Stochastic BayesFlow" for the 2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM). This repository is developed based on [BayesFlow](https://github.com/stefanradev93/BayesFlow) for the paper [BayesFlow: Learning complex stochastic models with invertible neural networks](https://arxiv.org/abs/2003.06281) and [stochastic normalizing flows](https://github.com/PaulLyonel/conditionalSNF) for the paper ["Stochastic Normalizing Flows for Inverse Problems: a Markov Chains Viewpoint"](https://epubs.siam.org/doi/10.1137/21M1450604). 
The main contributions of our work are as follows:
-- We propose stochastic BayesFlow as the extension of the original BayesFlow, contributing to avoiding overfitting to some extent with limited training data. 
-- We summarize and validate an algorithm for solving SIPs with (stochastic) BayesFlow using the inverse uncertainty quantification of a single-track vehicle model.
-- We also show that the stochastic BayesFlow outperforms BayesFlow and BNN in terms of the accuracy and precision of parameter identification, even with noisy observed data.
The vehicle model is referred to [CommonRoad](https://commonroad.in.tum.de/). The generated data is saved in /data.
We adapt the original [FrEIA](https://github.com/vislearn/FrEIA) and the original [conditionalSNF](https://github.com/PaulLyonel/conditionalSNF), and the original [LSTNet](https://github.com/laiguokun/LSTNet) to our use case.
The BayesFlow model is implemented in the script <Bayesian_conditional_normalizing_flow_LSTNet_dropout_cinn.py> and the stochastic BayesFlow model is implemented in the script <Bayesian_stochastic_conditional_normalizing_flow_MCMC.py>
