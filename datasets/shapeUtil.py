import matplotlib.pyplot as plt
import trimesh
import pyrender
import numpy as np
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def compute_camera_pose(camera_position, target_position, up_vector):
    forward = np.array(target_position) - np.array(camera_position)
    forward /= np.linalg.norm(forward)

    right = np.cross(up_vector, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position
    return pose

def shape2Img(path, numViews=8, image_size=(224, 224), orbitRadius=2.0, center=None):
    mesh = trimesh.load(path)
    if mesh.is_empty:
        raise ValueError(f"Could not load a valid mesh from {path}.")

    scale = 1.0 / max(mesh.extents)
    mesh.apply_scale(scale)

    if center is None:
        center = mesh.bounding_box.centroid

    renderMesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene = pyrender.Scene(ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))
    scene.add(renderMesh)

    light_intensity = 10.0
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity * 0.5)
    scene.add(key_light, pose=np.eye(4))
    scene.add(fill_light, pose=np.eye(4))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_node = scene.add(camera)

    renderer = pyrender.OffscreenRenderer(viewport_width=image_size[0], viewport_height=image_size[1])

    angles = np.linspace(0, 2 * np.pi, numViews, endpoint=False)
    images = []

    for angle in angles:
        camera_position = [
            orbitRadius * np.sin(angle),
            orbitRadius * 0.5,
            orbitRadius * np.cos(angle)
        ]

        camera_pose = compute_camera_pose(
            camera_position=camera_position,
            target_position=center,
            up_vector=[0, 1, 0]
        )

        scene.set_pose(camera_node, pose=camera_pose)

        color, _ = renderer.render(scene)
        images.append(Image.fromarray(color))

    renderer.delete()

    return images
