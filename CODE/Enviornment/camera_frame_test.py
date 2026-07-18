import mujoco
from PIL import Image
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

camera = model.camera("emeet_c960_camera")
original_position = camera.pos.copy()
original_position[0]= original_position[0]
original_position[1]= original_position[1]-0.04
original_position[2]= original_position[2]
camera.pos[:] = original_position
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)



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
Image.fromarray(image).save(Path(__file__).resolve().parent / "camera_view.png")

renderer.close()



