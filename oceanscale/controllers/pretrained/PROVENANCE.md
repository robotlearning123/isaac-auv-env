# Pretrained Model Provenance

## warpauv_poshold_ppo.pt
- **Source**: isaac-auv-env (https://github.com/warplab/isaac-auv-env)
- **Original path**: weights/2024-09-13_20-15-03_poshold_DR_2.pt
- **License**: BSD-3-Clause
- **Paper**: Lofaro et al., "Learning to Swim", arXiv 2410.00120
- **Task**: Position hold with domain randomization
- **Vehicle**: WarpAUV (6 thrusters, 22.7 kg)
- **Algorithm**: PPO (RSL-RL), 400 iterations, [64,64] MLP
- **Format**: PyTorch checkpoint dict with keys: model_state_dict, optimizer_state_dict, iter, infos
