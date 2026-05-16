export default {
  title: "Five constraints holding marine robotics back",
  problems: [
    {
      title: "Real ocean trials cost so much they cap iteration",
      body: "A day at sea: ¥10k–¥100k+. Corrosion, entanglement, gear loss. Non-reproducible currents. Progress gated by weather. Without high-fidelity simulation, iteration speed is locked to nature's clock.",
    },
    {
      title: "CFD is too slow for RL to even start",
      body: "Hi-fi CFD: 1 second of physics takes 1–100 hours to compute. The 1M+ episodes RL needs is mathematically unreachable. Even GPU-accelerated CFD can't approach the real-time × 1000× threshold the field actually requires.",
    },
    {
      title: "Legacy underwater sims waste modern GPUs",
      body: "Gazebo / UWSim / HoloOcean: single-threaded CPU at hundreds of FPS. RTX 5090 / H100 compute is wasted. Multi-agent parallelism is near zero.",
    },
    {
      title: "The sim-to-real gap kills policy transfer",
      body: "Insufficient fluid fidelity — sim policies collapse in water. No principled sim-to-real calibration. Sensor simulation (sonar, underwater camera) is oversimplified, creating false confidence.",
    },
    {
      title: "AI-native interfaces are missing",
      body: "Gym/gymnasium adapters absent or low-quality. Poor multi-agent scene support. Multimodal marine datasets (sonar, optical, IMU) aren't open — foundation-model training is gated by data scarcity.",
    },
  ],
};
