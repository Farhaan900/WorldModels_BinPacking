from gym.envs.registration import register

register(
    id='BinPackingEnv-v0',
    entry_point='bin_packing.bin_packing_environment:BinPackingGymEnvironment'
)

