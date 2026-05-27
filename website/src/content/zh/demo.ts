export default {
  eyebrow: "Live Demo",
  title: "从源码运行 BlueROV2 悬停 demo",
  subhead: "OceanScale 在 Newton+Warp GPU 上运行水下机器人动力学，并以 Isaac Sim 6 / Isaac Lab 3 作为验证线。",
  installCommand: "uv sync --extra dev && uv run oceanscale demo bluerov2-hover --device cpu",
  installLabel: "安装 & 运行",
  colabLabel: "在 Colab 中运行",
  colabUrl: "https://github.com/robotlearning123/oceanscale/blob/main/notebooks/bluerov2_hover_colab.ipynb",
  videoSrc: "/videos/bluerov2-demo.mp4",
  benchHeaders: ["环境数", "吞吐量 (env-steps/s)", "基准步数", "验证线"],
  benchRows: [
    ["n=1", "1,326", "200", "OceanScale"],
    ["n=64", "85,824", "200", "OceanScale"],
    ["n=4096", "4,588,922", "200", "OceanScale"],
  ],
  benchNote: "当前发布标准化基准为 RTX 5090 上的 OceanScale-only 结果；旧 PyBullet 对比单独保留在 benchmarks/RESULTS.md。",
  caveat: "v0.1 alpha — CPU demo 是 CLI smoke test，不是策略质量 gate。Isaac Sim 6 / Isaac Lab 3 验证由仓库 verifier 跟踪。",
};
