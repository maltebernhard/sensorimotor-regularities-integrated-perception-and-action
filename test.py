from components.aicon import MinimalAICON

# ==================================================================================

if __name__ == "__main__":

    aicon = MinimalAICON(num_obstacles = 2, internal_vel = True)

    seed = 10

    aicon.run(250, seed, render=True, prints=1, step_by_step=False, record_video=False)