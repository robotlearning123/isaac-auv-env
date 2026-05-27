"""Run all 7 OceanScale sensors on Isaac Sim 6."""
import sys
sys.path.insert(0, "/home/robot/workspace/46-marine")

import numpy as np
import warp as wp
wp.init()

# Build test mesh (flat seabed at z=-30)
verts = np.array([[-50,-50,-30],[50,-50,-30],[-50,50,-30],[50,50,-30]], dtype=np.float32)
indices = np.array([0,2,1,1,2,3], dtype=np.int32)
mesh = wp.Mesh(
    points=wp.array(verts, dtype=wp.vec3, device="cuda:0"),
    indices=wp.array(indices, dtype=wp.int32, device="cuda:0"),
)

pos = np.array([0, 0, -5], dtype=np.float32)
quat = np.array([0, 0, 0, 1], dtype=np.float32)

# 1. DVL
from oceanscale.sensors.ray_dvl import RayDVL
dvl = RayDVL(seabed_mesh=mesh, device="cuda:0")
dvl_data = dvl.measure(pos, quat)
print(f"DVL: altitude={dvl_data['altitude']:.1f}m, ranges={dvl_data['beam_ranges']}")

# 2. Imaging Sonar
from oceanscale.sensors.imaging_sonar import ImagingSonar, ImagingSonarConfig
sonar = ImagingSonar(environment_mesh=mesh, device="cuda:0")
img = sonar.scan(pos, quat)
print(f"ImagingSonar: shape={img.shape}")

# 3. Multibeam
from oceanscale.sensors.multibeam import MultibeamSonar, MultibeamConfig
mbes = MultibeamSonar(environment_mesh=mesh, device="cuda:0")
mbes_data = mbes.scan(pos, quat)
print(f"MBES: ranges={mbes_data['ranges'].shape}, intensity={mbes_data['intensity'].shape}")

# 4. Sidescan
from oceanscale.sensors.sidescan_sonar import SideScanSonar
sss = SideScanSonar(environment_mesh=mesh, device="cuda:0")
sss_data = sss.ping(pos, quat)
print(f"SSS: waterfall={sss_data.shape}")

# 5. Magnetometer
from oceanscale.sensors.magnetometer import Magnetometer
mag = Magnetometer()
mag_data = mag.measure(quat)
print(f"Magnetometer: field={mag_data} uT")

# 6. Acoustic Modem
from oceanscale.sensors.acoustic_modem import AcousticModem
modem = AcousticModem()
sender = np.array([0, 0, -5], dtype=np.float32)
receiver = np.array([100, 0, -10], dtype=np.float32)
pkt = modem.send(sender, receiver)
print(f"AcousticModem: TL={pkt['transmission_loss_db']:.1f}dB, delay={pkt['delay_s']*1000:.1f}ms, received={pkt['received']}")

# 7. USBL
from oceanscale.sensors.usbl import USBL
usbl = USBL()
usbl.add_transponder(np.array([10, 20, -15], dtype=np.float32), transponder_id=1)
fixes = usbl.measure(np.array([0, 0, -5], dtype=np.float32))
fix = fixes[0]
bearing_deg = float(np.degrees(fix['bearing_rad']))
print(f"USBL: range={fix['range_m']:.1f}m, bearing={bearing_deg:.1f}deg, valid={fix['valid']}")

print("\nALL 7 SENSORS PASSED ON ISAAC 6")
