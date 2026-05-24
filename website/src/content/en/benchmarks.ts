export default {
  eyebrow: "Early Benchmarks",
  title: "GPU speed meets underwater fidelity.",
  headers: ["Capability", "Classic Sims", "GPU Sims", "OceanScale"],
  rows: [
    ["GPU-parallel envs", "—", "✓", "**✓ (Newton+Warp)**"],
    ["Fossen hydrodynamics", "✓", "Weak", "**✓ (GPU kernel)**"],
    ["DVL / IMU / pressure", "✓", "Partial", "**✓ (stub + GPU)**"],
    ["RL-ready (Gymnasium)", "Difficult", "Partial", "**Native**"],
    ["Throughput (n=64)", "~800 FPS", "250K+", "**85,824 env-steps/s**"],
  ],
  landscapeNote: "Classic sims: Stonefish, HoloOcean, DAVE. GPU sims: MarineGym, OceanSim.",
  caveat: "OceanScale figure measured on RTX 5090, Torch CUDA 12.8, BlueROV2 hover task (2026-05-24). Other figures from published papers and public documentation.",
};
