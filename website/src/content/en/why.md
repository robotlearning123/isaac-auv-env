---
title: "Why OceanScale exists"
problems:
  - title: "Real ocean trials cost so much they cap iteration rate"
    body: "A day at sea: ¥10,000–¥100,000+ (ship time + crew + equipment depreciation). High equipment loss from corrosion, pressure, entanglement, loss. Non-reproducible conditions (currents, visibility, marine life). Progress gated by weather and season. Without high-fidelity simulation, iteration speed is locked to nature's clock."
  - title: "CFD is too slow for RL training to even start"
    body: "High-fidelity CFD: 1 second of physics takes 1–100 hours to compute. The 1M+ episodes RL training demands is mathematically unreachable. Even GPU-accelerated CFD cannot approach the real-time × 1000x threshold the field actually needs."
  - title: "Legacy underwater sims waste modern GPUs"
    body: "Gazebo / UWSim / HoloOcean: single-threaded CPU, tens to hundreds of FPS. Cannot fully utilize RTX 5090 / H100 compute. Multi-agent and multi-scene parallelism is poor."
  - title: "The sim-to-real gap still gates policy transfer"
    body: "Insufficient fluid dynamics fidelity → policies that work in sim collapse in water. No principled sim-to-real calibration method exists. Sensor simulation (sonar, underwater camera) is often oversimplified, creating false confidence."
  - title: "AI-native interfaces are missing — ecosystem hasn't caught up"
    body: "Gym/gymnasium adapters missing or low quality. Poor multi-agent scene support. Multimodal marine datasets (sonar, optical, IMU) are not openly available — foundation-model training is gated by data scarcity."
---
