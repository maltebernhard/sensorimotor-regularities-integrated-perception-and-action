from experiment_general.aicon import GeneralTestAICON, SimpleVelTestAICON

# ===========================================================================================

# 1 - General Test AICON
# 2 - Simple Vel Test AICON
test_id = 2
seed = 10

if __name__ == "__main__":
    if test_id == 1:
        print("Running General AICON Test")
        aicon = GeneralTestAICON(num_obstacles=0, internal_vel=False, vel_control=True)
        aicon.run(2500, seed, render=True, prints=1, step_by_step=True, record_video=False)

    if test_id == 2:
        print("Running Simple Vel Control AICON Test")
        aicon = SimpleVelTestAICON()
        aicon.run(2500, seed, render=True, prints=1, step_by_step=True, record_video=False)

    else:
        print("No Valid Test ID")