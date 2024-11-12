from components.aicon import MinimalAICON

# ==================================================================================

if __name__ == "__main__":

    aicon = MinimalAICON(0, False)
    # seed for one obstacle example | demonstrates estimation error when switching offset angle from pi/2 to -pi/2
    #seed = 19
    # seed for two obstacle example
    seed = 10

    for run in range(10):
        aicon.run(2000, seed+run, render=True, prints=1, step_by_step=False)