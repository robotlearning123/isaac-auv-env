export default {
  eyebrow: "Platform",
  title: "What OceanScale delivers",
  capabilities: [
    {
      metric: "11,435",
      metricLabel: "FPS · single-env BlueROV2 (RTX 5090)",
      title: "Full RL training pipeline",
      body: "Measured on RTX 5090 single card. End-to-end in simulation, environment to policy.",
    },
    {
      metric: "90 M",
      metricLabel: "env-steps/s @ 8192 envs",
      title: "Fossen multi-env throughput",
      body: "8192 parallel envs on Fossen 6-DoF kernel. 900× target achieved.",
    },
    {
      metric: "6-DoF",
      metricLabel: "Fossen + custom SPH",
      title: "Marine-specific physics kernels",
      body: "Rigid body dynamics + fluid simulation, purpose-built for ocean.",
    },
    {
      metric: null,
      metricLabel: "Newton + Isaac Sim 6",
      title: "GPU-native physics + render stack",
      body: "Newton engine jointly maintained by Anthropic + NVIDIA + Lightwheel + Apple.",
    },
    {
      metric: null,
      metricLabel: "gym / gymnasium native",
      title: "Drop-in RL interfaces",
      body: "Direct integration with rl_games / stable-baselines3 / RSL-RL.",
    },
    {
      metric: null,
      metricLabel: "Multi-agent · Procedural",
      title: "Real-world ocean asset library",
      body: "Ships / AUVs / ROVs / sensors + procedural ocean scenes.",
    },
  ],
};
