# aicon-contingencies

## ToDo:

- Covariance
    - Target Pos Estimate is in Robot Frame. Covariance Rotates with Robot because it is not transformed in forward model of estimator. --> How to approach this? If predict function is called multiple times, won't a direct reassignment of covariance lead to problems?

- Active Interconnection:
    - Specify class
        - one measurement model function or dependent on which state is selected?

- Estimators:
    - timestep

- Action gradient:
    - how to select / combine gradients?
        - consider PMoE / AICON gradient parallelism
        - encoding of individual goal / contingency action magnitude in: gradient magnitude vs. learned weights
            - consider timestep in AICON gradient computation
    - Vitos paper shows desirability of EE-movement on whole workspace: How is this possible? shouldn't there only be local gradient information avaiable?

## Implementation ToDo

- Measurement Model:


- Estimator


- Active Interconnection


- AICON class