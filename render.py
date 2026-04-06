import math
import os
import random
import time
import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from numba import jit
from progress import _load_checkpoint, _save_checkpoint

import numpy as np
from PIL import Image

from vec import *
from ray import *
from hittablelist import *
from sphere import *
from camera import *
from colour import *
from material import *

from scene_presets import *
from settings import *

def ray_col(r: Ray, bg_col: V3, world: Hittable, depth: int) -> V3:

    # If we've exceeded the ray bounce limit, no more light is gathered.
    if depth <= 0:
        return V3(0, 0, 0)

    # 0.001 gets rid of shadow acne
    has_hit, rec = world.hit(r, 0.001, math.inf)

    if not has_hit:
        return bg_col
    
    emitted = rec.material.emitted(rec.u, rec.v, rec.p)

    has_scatter, attenuation, scattered = rec.material.scatter(r, rec)
    if not has_scatter:
        return emitted
    
    return vec_add(emitted, vec_mul(attenuation, ray_col(scattered, bg_col, world, depth - 1)))

# Multiprocessing task
def render_scanline(scanline: list, world, cam, bg_col, samples: int, wt: int, ht: int, max_depth: int):
    j = scanline[0]
    rendered_scanline = []

    for i in scanline[1:]:
        sampled_col_sum = [0.0, 0.0, 0.0]
        for _ in range(samples):
            u = (i + random.random()) / (wt - 1)
            v = (ht - 1 - j + random.random()) / (ht - 1)
            ray_color = ray_col(cam.get_ray(u, v), bg_col, world, max_depth)
            sampled_col_sum[0] += ray_color.x
            sampled_col_sum[1] += ray_color.y
            sampled_col_sum[2] += ray_color.z

        sampled_v3 = V3(sampled_col_sum[0], sampled_col_sum[1], sampled_col_sum[2])
        rendered_scanline.append(compute_rgb_from_sample_sum(sampled_v3, samples))

    return j, rendered_scanline


def render_scene(world: HittableList, camera: Camera, settings: Settings):
    samples = settings.samples
    wt = settings.wt
    ht = settings.ht
    bg_col = settings.bg_col
    max_depth = settings.max_depth

    print('Setting up...')
    image_data, done = _load_checkpoint(settings, ht, wt)

    remaining = [j for j in range(ht) if not done[j]]
    if not remaining:
        print('Checkpoint already complete. Skipping render section.')
    else:
        coord_image = [[j] + list(range(wt)) for j in remaining]
        ntasks = len(coord_image)
        nprocs = cpu_count()

        print(f'Done loading. {done.sum()} / {ht} rows already completed.')
        print('\nRendering...')
        start = time.time()

        with Pool(nprocs) as p:
            m = p.imap_unordered(partial(render_scanline, world=world, cam=camera, bg_col=bg_col, samples=samples, wt=wt, ht=ht, max_depth=max_depth), coord_image)
            completed = int(done.sum())
            print(f'Render progress: {completed / ht:.2%}', end='\r')
            for sl in m:
                j, row = sl
                image_data[j] = np.array(row, dtype=np.uint8)
                done[j] = True
                completed += 1

                if completed % settings.checkpoint_interval == 0 or completed == ht:
                    _save_checkpoint(settings, image_data, done)

                print(f'Render progress: {completed / ht:.2%}   ', end='\r')

        dt = time.time() - start
        print('\nDone.')
        print(f'Render time ({nprocs} procs): {str(datetime.timedelta(seconds=dt))}')

    # Final save; if still in checkpoint mode, optionally keep checkpoint
    print('\nSaving image...')
    image = Image.fromarray(image_data)
    image.save(settings.output_path)
    print(f'Done. Saved {settings.output_path}')

    if settings.checkpoint_enabled and os.path.exists(settings.checkpoint_path):
        if done.all():
            print(f'Render completed. Checkpoint file remains: {settings.checkpoint_path}')
        else:
            print('Render aborted early, checkpoint saved for resume.')

    try:
        image.show()
    except Exception as e:
        print(f'Note: Could not open image viewer ({e}), but image was saved successfully.')