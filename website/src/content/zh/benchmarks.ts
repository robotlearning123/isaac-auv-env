export default {
  eyebrow: "早期基准",
  title: "GPU 速度与水下保真度的交汇。",
  headers: ["能力", "经典仿真", "GPU 仿真", "OceanScale"],
  rows: [
    ["GPU 并行环境", "—", "✓", "**✓ (Newton+Warp)**"],
    ["Fossen 水动力", "✓", "弱", "**✓ (GPU kernel)**"],
    ["DVL / IMU / 压力", "✓", "部分", "**✓ (stub + GPU)**"],
    ["RL 就绪 (Gymnasium)", "困难", "部分", "**原生**"],
    ["吞吐量 (n=64)", "~800 FPS", "250K+", "**17,427 env-steps/s**"],
  ],
  landscapeNote: "经典仿真: Stonefish, HoloOcean, DAVE。GPU 仿真: MarineGym, OceanSim。",
  caveat: "OceanScale 数据来自 RTX 5090, CUDA 12.9, BlueROV2 悬停任务 (2026-05-22)。其他数据来自已发表论文与公开文档。",
};
