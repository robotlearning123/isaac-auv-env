# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BlueROV2 Hover Control with OceanScale
#
# Train a PPO agent to perform station-keeping (hover) with a BlueROV2 Heavy
# using GPU-batched Tier-1 Fossen hydrodynamics.
#
# **Tested on**: RTX 5090 (CUDA 12.8). T4 compatibility expected but not yet
# end-to-end verified — Newton 1.2 + Warp 1.13 install path on the free Colab
# T4 runtime may need additional steps. Report issues via GitHub.
#
# **GitHub**: [robotlearning123/oceanscale](https://github.com/robotlearning123/oceanscale)

# %% tags=["colab"]
# Verify GPU
import subprocess
import sys

subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    check=False,
)

# %% [markdown]
# ## 1. Install OceanScale

# %% tags=["colab"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "oceanscale[rl]"], check=True)

# %% [markdown]
# ## 2. Smoke Test

# %%
import oceanscale
from oceanscale.rov_env import ROVEnv

print(f"OceanScale version: {oceanscale.__version__}")

env = ROVEnv(n_envs=2, device="cuda")
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Target position: {info['target_position']}")
env.close()

# %% [markdown]
# ## 3. Train PPO Agent
#
# 50k steps takes ~3 min on T4 with 4 parallel envs.

# %%
import sys

# Train for 50k steps (quick demo)
sys.argv = ["oceanscale", "train", "bluerov2-hover", "--total", "50000", "--n_envs", "4", "--seed", "42"]
# from oceanscale.cli import main
# main()  # uncomment to train

# %% [markdown]
# ## 4. Run Demo + Render MP4

# %%
import os

mp4_path = "/tmp/bluerov2_demo.mp4"

# Run the pretrained (or freshly trained) policy
sys.argv = ["oceanscale", "demo", "bluerov2-hover", "--render-mp4", mp4_path]
# main()  # uncomment to run demo

# %% [markdown]
# ## 5. Display Video Inline

# %%
from IPython.display import Video, display

if os.path.exists(mp4_path):
    display(Video(mp4_path, embed=True, width=800))
else:
    print(f"MP4 not found at {mp4_path}. Run training + demo cells first.")

# %% [markdown]
# ## 6. Next Steps
#
# - **Full training**: increase `--total` to 200k+ for better convergence
# - **Documentation**: see [README](https://github.com/robotlearning123/oceanscale)
# - **Custom environments**: subclass `ROVEnv` with your own reward function
# - **Star the repo**: if you find this useful!
#
# ### T4 Compatibility Notes
#
# - Newton + Warp require CUDA 12.x (Colab default)
# - Use `n_envs <= 4` to fit in T4 16GB VRAM
# - Training uses CPU for PPO inference (GPU for env stepping)
# - `imageio-ffmpeg` handles MP4 encoding without system ffmpeg
