export default {
  title: "Why marine robotics needs purpose-built simulation infrastructure",
  problems: [
    {
      title: "Real ocean trials cap iteration speed",
      body: "A day at sea: ¥10k–¥100k+. Corrosion, entanglement, gear loss. Currents and weather can't be reproduced. Iteration speed is bounded by nature, not by engineering capacity.",
    },
    {
      title: "Classical CFD can't reach RL training scale",
      body: "High-fidelity CFD: 1 second of physics takes 1–100 hours to compute. The 1M+ episodes RL requires is mathematically unreachable. Even GPU-accelerated CFD falls short of the real-time × 1000× threshold marine robotics RL actually needs.",
    },
    {
      title: "Legacy underwater sims waste modern GPUs",
      body: "Gazebo, UWSim, HoloOcean: single-threaded CPU at hundreds of FPS. RTX 5090 / H100 compute is left idle. Multi-agent parallelism is near zero.",
    },
    {
      title: "The sim-to-real gap kills policy transfer",
      body: "Fluid fidelity is insufficient — policies trained in sim collapse in water. No principled sim-to-real calibration. Sensor models (sonar, underwater camera) are oversimplified, creating false confidence.",
    },
    {
      title: "AI-native interfaces are missing",
      body: "Gym and gymnasium adapters are absent or low-quality. Multi-agent scene support is poor. Multimodal marine datasets (sonar, optical, IMU) aren't open — foundation-model training is gated by data scarcity.",
    },
  ],
};
