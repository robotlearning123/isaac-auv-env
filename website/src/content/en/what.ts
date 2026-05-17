export default {
  eyebrow: "Platform",
  title: "How OceanScale builds the GPU-native simulation engine for marine robotics",
  capabilities: [
    {
      metric: "11,435",
      metricLabel: "FPS · single-env BlueROV2 (RTX 5090)",
      title: "Full RL training pipeline",
      body: "End-to-end on a single RTX 5090. Environment-to-policy training runs entirely in simulation, no ocean trial required.",
    },
    {
      metric: "90 M",
      metricLabel: "env-steps/s @ 8192 envs",
      title: "Fossen multi-env throughput",
      body: "8192 parallel environments on the Fossen 6-DoF kernel. 900× the throughput required for production-scale RL, on a single GPU.",
    },
    {
      metric: "6-DoF",
      metricLabel: "Custom Fossen + SPH",
      title: "Marine-specific physics",
      body: "Rigid body dynamics and SPH fluid simulation, custom-built for the ocean. Replaces generic CFD with kernels tuned for marine robotics scale.",
    },
    {
      metric: null,
      metricLabel: "Newton + Isaac Sim 6",
      title: "GPU-native physics + render",
      body: "Newton physics, co-maintained by Anthropic, NVIDIA, Lightwheel, and Apple. Isaac Sim 6 photorealistic rendering. The full GPU-native stack.",
    },
    {
      metric: null,
      metricLabel: "gym / gymnasium native",
      title: "Drop-in RL interfaces",
      body: "Native gym and gymnasium compatibility. rl_games, stable-baselines3, and RSL-RL integrate without adapter code.",
    },
    {
      metric: null,
      metricLabel: "Multi-agent · Procedural",
      title: "Real-world ocean scenes",
      body: "Ship, AUV, ROV, and sensor assets paired with procedural ocean generation. The world layer for marine robot training.",
    },
  ],
};
