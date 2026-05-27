import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-WarpAUV-Direct-v1",
    entry_point=f"{__name__}.warpauv_env:WarpAUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warpauv_env:WarpAUVEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarpAUVPPORunnerCfg",
    },
)
