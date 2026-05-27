"""Sensor models for underwater vehicle simulation."""

from oceanscale.sensors.acoustic_modem import AcousticModem, AcousticModemConfig
from oceanscale.sensors.dvl import DVLSensor
from oceanscale.sensors.dvl_configs import DVLConfig
from oceanscale.sensors.imaging_sonar import ImagingSonar, ImagingSonarConfig
from oceanscale.sensors.imu import IMUSensor
from oceanscale.sensors.magnetometer import Magnetometer, MagnetometerConfig
from oceanscale.sensors.multibeam import MultibeamConfig, MultibeamSonar
from oceanscale.sensors.pressure import PressureSensor
from oceanscale.sensors.ray_dvl import RayDVL
from oceanscale.sensors.ray_sonar import RaySonar
from oceanscale.sensors.sidescan_sonar import SideScanSonar, SideScanSonarConfig
from oceanscale.sensors.underwater_camera import CameraConfig, UnderwaterCamera
from oceanscale.sensors.usbl import USBL, USBLConfig

__all__ = [
    "USBL",
    "AcousticModem",
    "AcousticModemConfig",
    "CameraConfig",
    "DVLConfig",
    "DVLSensor",
    "IMUSensor",
    "ImagingSonar",
    "ImagingSonarConfig",
    "Magnetometer",
    "MagnetometerConfig",
    "MultibeamConfig",
    "MultibeamSonar",
    "PressureSensor",
    "RayDVL",
    "RaySonar",
    "SideScanSonar",
    "SideScanSonarConfig",
    "USBLConfig",
    "UnderwaterCamera",
]
