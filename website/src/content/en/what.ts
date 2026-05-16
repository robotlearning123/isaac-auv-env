export default {
  eyebrow: "Platform",
  title: "What we're building",
  capabilities: [
    {
      metric: "11,435",
      metricLabel: "FPS · single-env BlueROV2 (RTX 5090)",
      title: "Full RL training pipeline",
      body: "End-to-end on a single RTX 5090. Environment to policy, all in simulation.",
    },
    {
      metric: "90 M",
      metricLabel: "env-steps/s @ 8192 envs",
      title: "Fossen multi-env throughput",
      body: "8192 parallel envs on the Fossen 6-DoF kernel. 900× target achieved.",
    },
    {
      metric: "6-DoF",
      metricLabel: "Custom Fossen + SPH",
      title: "Marine-specific physics",
      body: "Rigid body dynamics + fluid simulation, built for the ocean.",
    },
    {
      metric: null,
      metricLabel: "Newton + Isaac Sim 6",
      title: "GPU-native physics + render",
      body: "Newton engine maintained by Anthropic + NVIDIA + Lightwheel + Apple. Isaac Sim 6 rendering.",
    },
    {
      metric: null,
      metricLabel: "gym / gymnasium native",
      title: "Drop-in RL interfaces",
      body: "rl_games · stable-baselines3 · RSL-RL — direct integration.",
    },
    {
      metric: null,
      metricLabel: "Multi-agent · Procedural",
      title: "Real-world ocean scenes",
      body: "Ships / AUVs / ROVs / sensors + procedural ocean generation.",
    },
  ],
};
