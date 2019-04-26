# This script designed to apply watermarks to photos.
# Original code from: https://www.blog.pythonlibrary.org/2017/10/17/how-to-watermark-your-photos-with-python/
# Modifications made 4.26.2019 by Elizabeth Sheetz
# pip install pillow
# `#%%` - Signifies a runnable cell for use with `Jupyter Notebooks`
 
#%% Imports
from PIL import Image, ImageDraw, ImageFont
import time
#%% Variable Definitions:
IMG_PATH = 'kitten.jpg'
OUTPUT_PATH = 'kitten_watermarked_' + time.strftime("%Y%m%d-%H%M%S") + '.jpg'
WATERMARK_TEXT = 'sheetz'
FONT_FILE = "BERNHC.TTF"
#%% Make The Watermark:
def watermark_text(input_image_path,
                   output_image_path,
                   text, pos):
    photo = Image.open(input_image_path)

    # make the image editable
    drawing = ImageDraw.Draw(photo)

    black = (3, 8, 12)
    #font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
    font = ImageFont.truetype(FONT_FILE, size=50)
    drawing.text(pos, text, fill=black, font=font)
    photo.show()
    photo.save(output_image_path)

#%% Honestly dont know what this does haha
if __name__ == '__main__':
    watermark_text(IMG_PATH, OUTPUT_PATH,
                   text=WATERMARK_TEXT,
                   pos=(0, 0))
