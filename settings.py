from vec import *

class Settings:
    def __init__(self, image_wt: float, image_ht: float, bg_col: V3, samples: int, max_trace_depth: int, checkpoint_enabled: bool = True) -> None:
        self.wt = image_wt
        self.ht = image_ht
        self.bg_col = bg_col
        self.samples = samples
        self.max_depth = max_trace_depth

        self.aspect_ratio = self.wt / self.ht

        # checkpoint/resume settings
        self.checkpoint_enabled = checkpoint_enabled
        self.checkpoint_path = 'render_checkpoint.npz'
        self.checkpoint_interval = 10  # save every N rows
        self.output_path = 'out.png'