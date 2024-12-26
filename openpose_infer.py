from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
from PIL import Image
import os

input_dir = "./StoryMaker/traindata/original"
output_dir = "./StoryMaker/traindata/pose"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(input_dir, filename)
        image = Image.open(input_path)

        processor = OpenposeDetector.from_pretrained('./models/lllyasviel/ControlNet')
        control_image = processor(image, body_and_pose=True)

        output_path = os.path.join(output_dir, filename)
        control_image.save(output_path)
        print(f"Processed {filename} and saved to {output_path}")


