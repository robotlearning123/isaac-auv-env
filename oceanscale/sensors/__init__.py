"""Sensor models for underwater vehicle simulation."""

from oceanscale.sensors.acoustic_modem import AcousticModem, AcousticModemConfig
from oceanscale.sensors.dvl import DVLSensor
from oceanscale.sensors.imu import IMUSensor
from oceanscale.sensors.magnetometer import Magnetometer, MagnetometerConfig
from oceanscale.sensors.pressure import PressureSensor
from oceanscale.sensors.ray_dvl import RayDVL
from oceanscale.sensors.ray_sonar import RaySonar
from oceanscale.sensors.usbl import USBL, USBLConfig

__all__ = [
    "AcousticModem",
    "AcousticModemConfig",
    "DVLSensor",
    "IMUSensor",
    "Magnetometer",
    "MagnetometerConfig",
    "PressureSensor",
    "RayDVL",
    "RaySonar",
    "USBL",
    "USBLConfig",
]
