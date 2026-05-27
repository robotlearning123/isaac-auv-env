export default {
  eyebrow: "Live Demo",
  title: "Run BlueROV2 hover from the source checkout",
  subhead: "OceanScale runs underwater robot dynamics on Newton+Warp GPU and validates against the Isaac Sim 6 / Isaac Lab 3 lane.",
  installCommand: "uv sync --extra dev && uv run oceanscale demo bluerov2-hover --device cpu",
  installLabel: "Install & run",
  colabLabel: "Run in Colab",
  colabUrl: "https://github.com/robotlearning123/oceanscale/blob/main/notebooks/bluerov2_hover_colab.ipynb",
  videoSrc: "/videos/bluerov2-demo.mp4",
  benchHeaders: ["Envs", "Throughput (env-steps/s)", "Bench steps", "Gate"],
  benchRows: [
    ["n=1", "1,326", "200", "OceanScale"],
    ["n=64", "85,824", "200", "OceanScale"],
    ["n=4096", "4,588,922", "200", "OceanScale"],
  ],
  benchNote: "Launch-standardized OceanScale-only benchmark on RTX 5090. The older PyBullet comparison remains documented separately in benchmarks/RESULTS.md.",
  caveat: "v0.1 alpha — CPU demo is a CLI smoke test, not a policy-quality gate. Isaac Sim 6 / Isaac Lab 3 validation is tracked by the repo verifier.",
};
