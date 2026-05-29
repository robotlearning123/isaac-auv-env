# Real FloWave (Edinburgh) — Ground-Truth Reference

Authoritative spec of the **real** FloWave Ocean Energy Research Facility
(University of Edinburgh), for comparing against OceanScale's virtual FloWave.
Every number is cited. **Fleet agents may ONLY use values from this file (with
their sources); they must NOT assert Edinburgh facts from memory.** Items not
found here are gaps to flag, not to invent.

Retrieved 2026-05-29 via web search.

## Facility (cited)

| Quantity | Value | Source |
|---|---|---|
| Outer tank diameter | 30 m | edesign.co.uk portfolio; flowavett.co.uk |
| **Working** tank diameter | **25 m** | flowavett.co.uk/tank-design-and-features; edesign |
| Working water depth | **2.0 m** | en.wikipedia.org/wiki/FloWaveTT; flowavett |
| Wavemaker paddles | **168**, hinged **flap**, **active-absorbing** | Wikipedia; flowavett; edesign |
| Flow-drive units (current) | **28** pumps/propellers beneath floor | Wikipedia; flowavett/facility |
| Max current velocity | **1.6 m/s**, any direction (360°) | Wikipedia; flowavett |
| Water volume | 2.4 million litres fresh water | Wikipedia |
| Usable test area | ~15 m Ø central region | Wikipedia |
| Opened | June 2014 | Wikipedia |

## Wave capability (cited)

| Quantity | Value | Source |
|---|---|---|
| Paddle design point | **optimised for 0.7 m (700 mm) high, 2 s period waves** | edesign.co.uk; flowavett |
| Wave types | full-spectrum multi-directional + monochromatic | edesign; flowavett |
| Directionality | 360°, waves and current independent | flowavett; Wikipedia |
| Full-scale equivalent | up to ~28 m seas (at scale) | edesign |
| Active absorption | paddles push/pull to BOTH make AND absorb incoming waves (prevent reflections) | Wikipedia ("active-absorbing"); flowavett |

## Sources
- https://en.wikipedia.org/wiki/FloWaveTT
- https://www.flowavett.co.uk/tank-design-and-features
- https://www.flowavett.co.uk/facility
- https://flowave.eng.ed.ac.uk/how-it-works
- http://www4.edesign.co.uk/portfolio/edinburgh-university/
- https://eng.ed.ac.uk/about/facilities/flowave-ocean-energy-research-facility

## NOT FOUND (gaps in public spec — agents mark "(not found)", do not invent)
- Exact paddle width / hinge depth below SWL (numeric).
- Max regular wave height vs period curve (the Ingram/edesign operating envelope — paywalled/not located this session).
- Paddle max stroke & max velocity (mechanical limits).
- Exact wave frequency/period min–max range.
- Number/placement detail of the 28 impellers; turbulence intensity spec.

## OURS (OceanScale virtual FloWave — verified from code, the comparison baseline)
- `paddle_array.py FlapPaddleArray`: R=12.5 m (→25 m Ø), N=168 flap, h=2.0 m, hinge_depth=1.9 m, g=9.81; φ_n=2πn/168.
- Dispersion ω²=g·k·tanh(kh) (Newton). transfer_fn_HS = bottom-hinged flap `4·sinh(kh)(kh·sinh kh−cosh kh+1)/(kh(sinh2kh+2kh))` (→2 deep water). Snake principle (Biésel 1954). Irregular = Miles-Funke single-summation, a_j=√(2·S·Δω·Δθ).
- `impeller_array.py ImpellerArray`: basin_radius=12.5, uniform_radius=5.5, motor_max_rpm=200, max_uniform_speed=1.6 m/s, default_TI=0.07.
- `water_usd.py`: SWL=2.0 m, 256×256 surface grid over [-12,12]².
- Acceptance/render this session used H=0.1 m, T=2.0 s (sub-design wave; facility design point is 0.7 m @ 2 s).
- **Active wave absorption: NOT modelled** (ours is forward spectral synthesis only) — candidate top gap.
