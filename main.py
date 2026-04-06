from vec import *
from camera import *

from scene_presets import *
from settings import *
from render import render_scene

def main():
    # Image
    aspect_ratio = 1.0
    image_wt = 400
    image_ht = int(image_wt / aspect_ratio)
    samples_per_pixel = 5 # recommended: 100 or more for good quality, but this is just for testing
    max_depth = 5 # recommended: 50 or more for good quality, but this is just for testing
    bg_col = V3(0, 0, 0)
    
    # Decide whether to start from last checkpoint
    use_checkpoint =  False # Set to False to start fresh
    
    s = Settings(image_wt, image_ht, bg_col, samples_per_pixel, max_depth, use_checkpoint)

    # World
    world, cam = simple_scene()
    if cam is None:
        cam = Camera(V3(0,0,-5), V3(0,0,0), V3(0,1,0), 90, 1, 0, 5, 0, 1)

    render_scene(world, cam, s)

if __name__ == '__main__':
    main()