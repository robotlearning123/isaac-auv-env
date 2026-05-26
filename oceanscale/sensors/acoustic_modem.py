"""Underwater acoustic modem for inter-vehicle communication.

Models data-rate/range trade-off, propagation delay, and packet loss from
multipath and distance-dependent attenuation.  Uses transmission loss from
``oceanscale.acoustics`` when available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AcousticModemConfig:
    """Acoustic modem parameters (EvoLogics S2C R-based defaults)."""

    frequency_khz: float = 26.0
    bandwidth_khz: float = 16.0
    max_range: float = 3500.0
    data_rate_bps: float = 13900.0
    source_level_db: float = 185.0
    noise_level_db: float = 60.0
    packet_loss_base: float = 0.01
    sound_speed: float = 1500.0
    noise_seed: int | None = None


class AcousticModem:
    """Simulated underwater acoustic modem."""

    def __init__(self, config: AcousticModemConfig | None = None) -> None:
        cfg = config or AcousticModemConfig()
        self.cfg = cfg
        self._rng = np.random.RandomState(cfg.noise_seed)

    def send(
        self,
        sender_position: np.ndarray,
        receiver_position: np.ndarray,
        payload_bytes: int = 64,
    ) -> dict:
        """Simulate sending a packet from sender to receiver.

        Returns dict with keys: received, delay_s, data_rate_bps,
        transmission_loss_db, snr_db, range_m.
        """
        sender = np.asarray(sender_position, dtype=np.float32)
        receiver = np.asarray(receiver_position, dtype=np.float32)
        distance = float(np.linalg.norm(receiver - sender))

        delay = distance / self.cfg.sound_speed

        if distance < 0.1:
            spreading_loss = 0.0
        else:
            spreading_loss = 20.0 * math.log10(distance)

        alpha_db_per_km = 0.036 * self.cfg.frequency_khz ** 1.5
        absorption = alpha_db_per_km * distance / 1000.0
        tl = spreading_loss + absorption

        snr = self.cfg.source_level_db - tl - self.cfg.noise_level_db

        loss_rate = self.cfg.packet_loss_base
        if distance > self.cfg.max_range * 0.5:
            loss_rate += 0.2 * ((distance - self.cfg.max_range * 0.5) / (self.cfg.max_range * 0.5)) ** 2
        if distance > self.cfg.max_range:
            loss_rate = 1.0
        if snr < 5.0:
            loss_rate = min(loss_rate + 0.5 * (5.0 - snr) / 5.0, 1.0)

        received = bool(self._rng.random() > loss_rate)

        effective_rate = self.cfg.data_rate_bps * max(0.0, min(1.0, snr / 20.0))

        return {
            "received": received,
            "delay_s": float(delay),
            "data_rate_bps": float(effective_rate),
            "transmission_loss_db": float(tl),
            "snr_db": float(snr),
            "range_m": float(distance),
            "payload_bytes": payload_bytes,
        }
