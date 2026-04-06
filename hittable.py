from vec import *
from ray import *
from aabb import *

class HitRecord:
    def __init__(self) -> None:
        self.p = None
        self.normal = None
        self.material = None
        self.t: float = 0
        self.u: float = 0
        self.v: float = 0
        self.front_face: bool = False

    def set_face_normal(self, r: Ray, outward_normal: V3) -> None:
        self.front_face = vec_dot(r.d, outward_normal) < 0
        
        if self.front_face:
            self.normal = outward_normal
        else:
            self.normal = outward_normal.neg()

class Hittable:
    def __init__(self) -> None:
        pass

    def hit(self, r: Ray, t_min: float, t_max: float) -> tuple[bool, HitRecord]:
        pass

    def bounding_box(self, _time0: float, _time1: float) -> tuple[bool, AABB]:
        pass