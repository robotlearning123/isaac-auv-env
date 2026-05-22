export default {
  eyebrow: "Live Demo",
  title: "BlueROV2 悬停训练比 CPU 快 10.5 倍",
  subhead: "OceanScale 在 Newton+Warp GPU 上运行 Fossen 6-DOF 模型。悬停、对接、搜索 — 大规模并行。",
  installCommand: "pip install oceanscale && oceanscale demo bluerov2-hover --render-mp4 hover.mp4",
  installLabel: "安装 & 运行",
  colabLabel: "在 Colab 中运行",
  colabUrl: "https://github.com/robotlearning123/46-marine/blob/main/notebooks/bluerov2_hover_colab.ipynb",
  videoSrc: "/videos/bluerov2-demo.mp4",
  benchHeaders: ["环境数", "吞吐量 (steps/s)", "1M 步耗时", "加速比"],
  benchRows: [
    ["n=1 (PyBullet)", "1,661", "603 s", "1.0x"],
    ["n=16", "5,076", "197 s", "3.1x"],
    ["n=64", "17,427", "57 s", "10.5x"],
  ],
  benchNote: "实测于 RTX 5090, BlueROV2 悬停任务, 30K 步。详见 benchmarks/RESULTS.md。",
  caveat: "v0.1 alpha — 策略可命中目标深度，但 33 秒回合内横向漂移约 0.41m。完整 station-keeping 留待 v0.2。",
};
