import glob
import os
import sys

try:
    sys.path.append(glob.glob('./carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    raise RuntimeError("Couldn't import the CARLA egg from ./carla")

import carla

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

spectator = world.get_spectator()
transform = spectator.get_transform()

print("Location:", transform.location)
print("Rotation:", transform.rotation)
