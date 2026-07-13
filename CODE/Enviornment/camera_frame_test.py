import mujoco
import imageio
from pathlib import Path

world_xml = Path(__file__).resolve().parent / "xml_models" / "world.xml"

model = mujoco.MjModel.from_xml_path(str(world_xml))
data = mujoco.MjData(model)

# Compute body, geom, and camera world positions
mujoco.mj_forward(model, data)

camera_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    "emeet_c960_camera"
)

print("Camera ID:", camera_id)
print("Camera position:", data.cam_xpos[camera_id])
print("Camera orientation:", data.cam_xmat[camera_id].reshape(3, 3))

renderer = mujoco.Renderer(
    model,
    width=1920,
    height=1080
)

renderer.update_scene(
    data,
    camera=camera_id
)

image = renderer.render()
imageio.imwrite("camera_view.png", image)

renderer.close()