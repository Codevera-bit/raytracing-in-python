import os
import numpy as np
from settings import Settings

def _load_checkpoint(settings: Settings, ht: int, wt: int):
    if not settings.checkpoint_enabled:
        return np.zeros((ht, wt, 3), dtype=np.uint8), np.zeros(ht, dtype=bool)

    chk_path = settings.checkpoint_path
    if not os.path.exists(chk_path):
        return np.zeros((ht, wt, 3), dtype=np.uint8), np.zeros(ht, dtype=bool)

    try:
        data = np.load(chk_path)
        image_data = data['image_data']
        done = data['done']

        if image_data.shape != (ht, wt, 3) or done.shape != (ht,):
            print('Checkpoint shape does not match current settings. Ignoring checkpoint and starting fresh.')
            return np.zeros((ht, wt, 3), dtype=np.uint8), np.zeros(ht, dtype=bool)

        # Flip upside down to correct orientation (old checkpoints were inverted)
        # image_data = np.flipud(image_data)

        print(f'Resuming from checkpoint: {chk_path}. ({done.sum()}/{ht} rows complete)')
        return image_data, done

    except Exception as e:
        print(f'Could not read checkpoint file {chk_path} ({e}). Starting fresh.')
        return np.zeros((ht, wt, 3), dtype=np.uint8), np.zeros(ht, dtype=bool)
    
def _save_checkpoint(settings: Settings, image_data: np.ndarray, done: np.ndarray):
    if not settings.checkpoint_enabled:
        return
    chk_path = settings.checkpoint_path
    tmp_path = f'{chk_path}.tmp'
    np.savez_compressed(tmp_path, image_data=image_data, done=done)
    # numpy will append .npz to file name; use same behavior and move to final path
    final_tmp = tmp_path if tmp_path.endswith('.npz') else f'{tmp_path}.npz'
    if final_tmp != tmp_path:
        os.replace(final_tmp, chk_path)
    else:
        os.replace(tmp_path, chk_path)