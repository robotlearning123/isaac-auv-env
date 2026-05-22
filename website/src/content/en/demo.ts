export default {
  eyebrow: "Live Demo",
  title: "Train BlueROV2 hover 10.5x faster than CPU baseline",
  subhead: "OceanScale runs the Fossen 6-DOF model on Newton+Warp GPU. Vectorize hover, dock, search — at scale.",
  installCommand: "pip install oceanscale && oceanscale demo bluerov2-hover --render-mp4 hover.mp4",
  installLabel: "Install & run",
  colabLabel: "Run in Colab",
  colabUrl: "https://github.com/robotlearning123/46-marine/blob/main/notebooks/bluerov2_hover_colab.ipynb",
  videoSrc: "/videos/bluerov2-demo.mp4",
  benchHeaders: ["Envs", "Throughput (steps/s)", "Time for 1M steps", "Speedup"],
  benchRows: [
    ["n=1 (PyBullet)", "1,661", "603 s", "1.0x"],
    ["n=16", "5,076", "197 s", "3.1x"],
    ["n=64", "17,427", "57 s", "10.5x"],
  ],
  benchNote: "Measured on RTX 5090, BlueROV2 hover task, 30K steps. See benchmarks/RESULTS.md for methodology.",
  caveat: "v0.1 alpha — hover policy reaches depth target but drifts ~0.41m laterally over a 33s eval. Full station-keeping is v0.2.",
};
