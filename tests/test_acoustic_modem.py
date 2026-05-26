"""Tests for acoustic modem sensor."""

import numpy as np

from oceanscale.sensors.acoustic_modem import AcousticModem, AcousticModemConfig


def test_modem_creates():
    modem = AcousticModem()
    assert modem.cfg.max_range == 3500.0


def test_modem_send_close_range():
    modem = AcousticModem(AcousticModemConfig(noise_seed=42))
    result = modem.send(np.array([0, 0, 0]), np.array([10, 0, 0]))
    assert result["received"]
    assert result["range_m"] == 10.0
    assert result["delay_s"] > 0


def test_modem_propagation_delay():
    modem = AcousticModem(AcousticModemConfig(sound_speed=1500.0))
    result = modem.send(np.array([0, 0, 0]), np.array([1500, 0, 0]))
    assert abs(result["delay_s"] - 1.0) < 0.01


def test_modem_out_of_range():
    modem = AcousticModem(AcousticModemConfig(max_range=100.0, noise_seed=42))
    result = modem.send(np.array([0, 0, 0]), np.array([200, 0, 0]))
    assert not result["received"]


def test_modem_transmission_loss_increases():
    modem = AcousticModem(AcousticModemConfig(noise_seed=42))
    tl_near = modem.send(np.array([0, 0, 0]), np.array([10, 0, 0]))["transmission_loss_db"]
    tl_far = modem.send(np.array([0, 0, 0]), np.array([1000, 0, 0]))["transmission_loss_db"]
    assert tl_far > tl_near


def test_modem_snr_decreases_with_distance():
    modem = AcousticModem(AcousticModemConfig(noise_seed=42))
    snr_near = modem.send(np.array([0, 0, 0]), np.array([10, 0, 0]))["snr_db"]
    snr_far = modem.send(np.array([0, 0, 0]), np.array([1000, 0, 0]))["snr_db"]
    assert snr_near > snr_far


def test_modem_data_rate_positive():
    modem = AcousticModem(AcousticModemConfig(noise_seed=42))
    result = modem.send(np.array([0, 0, 0]), np.array([50, 0, 0]))
    assert result["data_rate_bps"] > 0


def test_modem_payload_preserved():
    modem = AcousticModem(AcousticModemConfig(noise_seed=42))
    result = modem.send(np.array([0, 0, 0]), np.array([10, 0, 0]), payload_bytes=256)
    assert result["payload_bytes"] == 256
