export default {
  eyebrow: "Early Benchmarks",
  title: "GPU speed meets underwater fidelity.",
  headers: ["Capability", "Classic Sims", "GPU Sims", "OceanScale"],
  rows: [
    ["GPU-parallel envs", "—", "✓", "**✓ (Newton+Warp)**"],
    ["Fossen hydrodynamics", "✓", "Weak", "**✓ (GPU kernel)**"],
    ["DVL / sonar sensors", "✓", "Partial", "**✓ (Warp)**"],
    ["RL-ready (Gymnasium)", "Difficult", "Partial", "**Native**"],
    ["Throughput (n=64)", "~800 FPS", "250K+", "**17,427 env-steps/s**"],
  ],
  landscapeNote: "Classic sims: Stonefish, HoloOcean, DAVE. GPU sims: MarineGym, OceanSim.",
  caveat: "OceanScale figures measured on RTX 5090, CUDA 12.9, BlueROV2 hover task (2026-05-22). Other figures from published papers and public documentation.",
};
