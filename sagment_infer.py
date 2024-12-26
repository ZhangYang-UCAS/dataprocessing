# from PIL import Image
# from segment import ObjectSegmenter
# segmenter = ObjectSegmenter()

# image = Image.open("./assets/avg_prompt/Hades_main.webp")
# mask = segmenter.segment(image, "person")
# mask.save("./assets/avg_prompt/Hades_mask.png")

import os
from PIL import Image
from segment import ObjectSegmenter
segmenter = ObjectSegmenter()

input_dir = "./StoryMaker/traindata/original"
output_dir = "./StoryMaker/traindata/segmentation"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(input_dir, filename)

        image = Image.open(input_path)
        mask = segmenter.segment(image, "person")
        output_path = os.path.join(output_dir, filename)
        mask.save(output_path)
        print(f"Processed {filename} and saved to {output_path}")