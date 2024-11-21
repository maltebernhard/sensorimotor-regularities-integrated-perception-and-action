from components.aicon import MinimalAICON

# ==================================================================================

if __name__ == "__main__":

    aicon = MinimalAICON(num_obstacles = 0, internal_vel = True)

    seed = 10

    aicon.run(2500, seed, render=True, prints=1, step_by_step=True, record_video=False)