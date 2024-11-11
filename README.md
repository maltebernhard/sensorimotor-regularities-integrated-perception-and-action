# aicon-contingencies

## ToDo:

- Meeting:
    - Covariance
        - Target Pos Estimate is in Robot Frame. Covariance Rotates with Robot because it is not transformed in forward model of estimator. --> How is covariance propagated?

    - Active Interconnection:
        - Specify class
            - one measurement model function or dependent on which state is selected?

    - Action gradient:
        - how to select / combine gradients?
        - Vitos paper shows desirability of EE-movement on whole workspace: How is this possible? shouldn't there only be local gradient information avaiable?

    - RLCPC vs. AICON
        - define actions according to contingencies (gradient decent along loss function of goal)
        - 






## Implementation ToDo

- Measurement Model:
    - How to deal with multiple measurements? Returning H_t as a dict leads to messed up jacobian computation

- Estimator


- Active Interconnection


- AICON class