import trimesh
import pyrender
import numpy as np
from PIL import Image

def shape2Img(path, numViews=8, image_size=(224, 224), orbitRadius=2.0, center=None):
    mesh = trimesh.load(path)
    if mesh.is_empty:
        raise ValueError(f"Could not load a valid mesh from {path}.")
    if center is None:
        center = mesh.bounding_box.centroid

    scene = pyrender.Scene()
    renderMesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(renderMesh)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    camera_node = scene.add(camera, pose=np.eye(4))

    renderer = pyrender.OffscreenRenderer(viewport_width=image_size[0], viewport_height=image_size[1])

    angles = np.linspace(0, 2 * np.pi, numViews, endpoint=False)
    baseTranslation = np.eye(4, dtype=np.float32)
    baseTranslation[:3, 3] = -np.array(center)

    images = []
    for angle in angles:
        rotation = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        translation = np.array([[1, 0, 0, orbitRadius],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=np.float32)
        flipX = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        cameraPose = baseTranslation @ rotation @ translation @ flipX

        scene.set_pose(camera_node, pose=cameraPose)

        color, _ = renderer.render(scene)
        images.append(Image.fromarray(color))

    renderer.delete()
    
    return images
