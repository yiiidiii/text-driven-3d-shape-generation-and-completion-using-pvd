import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import numpy as np
import bpy
from mathutils import Vector
import os
from torch.nn.functional import cosine_similarity
import clip
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class ModelCaptionGenerator:
    def __init__(self, model_type='pretrain_flant5xxl', use_qa=False, local_checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.use_qa = use_qa
        self.local_checkpoint_path = local_checkpoint_path
        self.setup_model()

    def setup_model(self):
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name='blip2_t5',
            model_type=self.model_type,
            is_eval=True,
            device=self.device
        )
        if self.local_checkpoint_path and os.path.exists(self.local_checkpoint_path):
            self.model.load_checkpoint(self.local_checkpoint_path)
        else:
            print("Local checkpoint path is invalid or does not exist. Proceeding with the pre-trained model.")

    def setup_blender_scene(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        
        bpy.context.scene.camera.location = Vector((0, -3, 1.5))
        bpy.context.scene.camera.data.clip_start = 0.1
        bpy.context.scene.camera.data.clip_end = 1000
        
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 64

        # Three-point lighting
        self.add_light("KeyLight", energy=1000, location=(4, -4, 4))
        self.add_light("FillLight", energy=300, location=(-4, -4, 2))
        self.add_light("RimLight", energy=600, location=(0, 4, 4))

    def add_light(self, name, energy, location):
        bpy.ops.object.light_add(type='AREA', location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy

    def normalize_and_center_object(self):
        # Center and scale
        objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        bbox_min = Vector((float('inf'), float('inf'), float('inf')))
        bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))
        
        for obj in objects:
            for vertex in obj.bound_box:
                coord = obj.matrix_world @ Vector(vertex)
                bbox_min = Vector(map(min, bbox_min, coord))
                bbox_max = Vector(map(max, bbox_max, coord))
        
        center = (bbox_min + bbox_max) / 2
        dimensions = bbox_max - bbox_min
        max_dim = max(dimensions)
        
        for obj in objects:
            obj.location -= center
            obj.scale /= max_dim

        bpy.context.view_layer.update()

    def generate_views(self, model_path):
        self.setup_blender_scene()
        bpy.ops.wm.ply_import(filepath=model_path)

        self.normalize_and_center_object()
        views = []

        # 8 views around the object
        camera = bpy.context.scene.camera
        for i in range(8):
            angle = i * 45  # 45 degrees between each view
            camera.location.x = 2.0 * np.cos(np.radians(angle))
            camera.location.y = 2.0 * np.sin(np.radians(angle))
            camera.location.z = 1.5

            direction = Vector((0, 0, 0)) - camera.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()

            temp_path = f"/tmp/render_view_{i}.png"
            bpy.context.scene.render.filepath = temp_path
            bpy.ops.render.render(write_still=True)
            
            views.append(Image.open(temp_path).convert("RGB"))

        return views

    def rank_and_select_captions(self, images, captions):
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
        ranked_captions = []

        for image, caption_set in zip(images, captions):
            image_tensor = clip_preprocess(image).unsqueeze(0).to(self.device)
            text_tensors = clip.tokenize(caption_set).to(self.device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image_tensor)
                text_features = clip_model.encode_text(text_tensors)

            similarities = cosine_similarity(image_features, text_features).cpu().numpy()
            best_caption = caption_set[np.argmax(similarities)]
            ranked_captions.append(best_caption)

        return ranked_captions

    def consolidate_captions(self, captions, views):
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        total_similarities = np.zeros(len(captions))
        for view in views:
            image_tensor = clip_preprocess(view).unsqueeze(0).to(self.device)
            text_tensors = clip.tokenize(captions).to(self.device)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(image_tensor)
                text_features = clip_model.encode_text(text_tensors)
                similarities = cosine_similarity(image_features, text_features).cpu().numpy()
                total_similarities += similarities[0]
        
        avg_similarities = total_similarities / len(views)
        sorted_pairs = sorted(zip(captions, avg_similarities), key=lambda x: x[1], reverse=True)
        sorted_captions = [cap for cap, _ in sorted_pairs]
        
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir).to(self.device)

        input_text = " | ".join(sorted_captions)
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=min(input_ids.size(1) + 50, 512),
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=2,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_captions(self, views):
        all_captions = []
        for image in views:
            processed_image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
            try:
                if self.use_qa:
                    prompt = "Question: what object is in this image? Answer:"
                    object_name = self.model.generate({"image": processed_image, "prompt": prompt})[0]
                    full_prompt = f"Question: what is the structure and geometry of this {object_name}?"
                    captions = self.model.generate(
                        {"image": processed_image, "prompt": full_prompt},
                        num_captions=3
                    )
                else:
                    captions = self.model.generate(
                        {"image": processed_image},
                        num_captions=3
                    )
                
                captions = [str(cap) for cap in captions if cap]
                
                if not captions:
                    captions = ["No caption generated"]
                
                all_captions.append(captions)
            
            except Exception as e:
                print(f"Caption generation error: {e}")
                all_captions.append(["Failed to generate caption"])

        return all_captions

    def process_model(self, model_path):
        views = self.generate_views(model_path)
        raw_captions = self.generate_captions(views)
        ranked_captions = self.rank_and_select_captions(views, raw_captions)
        final_caption = self.consolidate_captions(ranked_captions, views)
        
        return final_caption.split(" | ")