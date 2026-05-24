"""Convert oceanscale/data/vec_normalize.pkl to vec_normalize.npz.

Run once at packaging time to produce the safe .npz file, then delete vec_normalize.pkl.
The .npz replaces pickle-based VecNormalize loading to remove the RCE risk from
arbitrary __reduce__ payloads in a tampered wheel.

Usage:
    python scripts/convert_vec_normalize_pkl_to_npz.py
"""

import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    data_dir = Path(__file__).parent.parent / "oceanscale" / "data"
    pkl_path = data_dir / "vec_normalize.pkl"
    npz_path = data_dir / "vec_normalize.npz"

    with open(pkl_path, "rb") as f:
        vn = pickle.load(f)  # noqa: S301 — one-time conversion script only

    np.savez(
        npz_path,
        obs_mean=vn.obs_rms.mean.astype(np.float64),
        obs_var=vn.obs_rms.var.astype(np.float64),
        obs_count=np.array(vn.obs_rms.count, dtype=np.float64),
        clip_obs=np.array(vn.clip_obs, dtype=np.float64),
        epsilon=np.array(vn.epsilon, dtype=np.float64),
    )
    print(f"Wrote {npz_path}")
    print(f"  obs_mean shape : {vn.obs_rms.mean.shape}")
    print(f"  obs_var  shape : {vn.obs_rms.var.shape}")
    print(f"  obs_count      : {vn.obs_rms.count}")
    print(f"  clip_obs       : {vn.clip_obs}")
    print(f"  epsilon        : {vn.epsilon}")


if __name__ == "__main__":
    main()
