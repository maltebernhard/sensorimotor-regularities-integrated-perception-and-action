import torch
from components.specifics import Target_Pos_MM, Target_Pos_Estimator

# ==================================================================================================

def print_state(msg, state):
    if type(state) == tuple:
        print(msg)
        print(state[0])
        print(state[1])
    else:
        print(msg)
        print(state)

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu") # Force CPU
    ###########################################################

    target_pos_measurement_model = Target_Pos_MM().to(device)
    target_distance_estimator = Target_Pos_Estimator().to(device)

    # for motion model: vel_frontal, vel_lateral, vel_rot, del_t
    us = torch.tensor([
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0.],
        [0.1, 0.1, 0.1, 0.1, 0.1]
    ]).T.to(device)
    # for measurement model: angular offsets to target
    z1s = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]).to(device)

    print_state('Multiple measurements state init:', target_distance_estimator.state)
    print("==============================================")

    for u, z1 in zip(us, z1s):
        print(f"u: {u}\ntype: {u.dtype}")
        target_distance_estimator.predict(u)
        
        print_state("Before Measurement", target_distance_estimator.state)
        target_distance_estimator.update_with_specific_meas({'target_offset_angle': z1}, target_pos_measurement_model)
        print_state("After Measurement", target_distance_estimator.state)
        print("==============================================")