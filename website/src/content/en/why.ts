export default {
  eyebrow: "Challenge",
  title: "The bottleneck is iteration speed.",
  label: "Iteration cost / underwater robotics",
  problems: [
    {
      title: "Sea windows",
      body: "Real ocean conditions cannot be queued like compute jobs.",
    },
    {
      title: "Physics cost",
      body: "High-fidelity fluid solvers are too expensive for policy search loops.",
    },
    {
      title: "Training scale",
      body: "Legacy simulators were not built for 8192 environments or RL workloads.",
    },
  ],
};
