from PIL import Image
import os

dir = "images"

for filename in os.listdir(dir):
    if filename.endswith(".jpg"):
        os.system(f"mv {dir}/{filename} {dir}/{filename[:-4]}.JPG")
        continue
    else:
        continue
