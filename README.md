# aicon-contingencies

## ToDo:

- Meeting:
    - Covariance
        - Target Pos Estimate is in Robot Frame. Covariance Rotates with Robot because it is not transformed in forward model of estimator. --> How is covariance propagated? 
        - Robot Vel estimate covariance becomes large when there is one measurement model for all 3 dimensions. If there is 3 individual measurement models, the estimate becomes better.
            - Shouldn't the local linearization of the unified measurement model equally show how the state dimensions relate to the measurement?
    - Active Interconnection:
        - Should I model measurement models individually now?
    - Action gradient:
        - Vitos paper shows desirability of EE-movement on whole workspace: How is this possible? shouldn't there only be local gradient information avaiable?







## Implementation ToDo

- Measurement Model:
    - How to deal with multiple measurements? Returning H_t as a dict leads to messed up jacobian computation

- Estimator


- Active Interconnection


- AICON class