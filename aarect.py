from hittable import *
from material import *

class XYrect(Hittable):
    """
    Represents a rectangle in the XY plane at a fixed Z depth (k).
    Defined by bounds [x0, x1] and [y0, y1].
    """
    def __init__(self, _x0: float, _x1: float, _y0: float, _y1: float, _k: float, _mat: Material) -> None:
        self.x0 = _x0
        self.x1 = _x1
        self.y0 = _y0
        self.y1 = _y1
        self.k = _k # The fixed Z-coordinate
        self.mat = _mat

    def hit(self, r: Ray, t_min: float, t_max: float) -> tuple[bool, HitRecord]:
        # Solve for t: o.z + t * d.z = k  =>  t = (k - o.z) / d.z
        t = (self.k - r.o.z) / r.d.z
        
        # Check if the intersection point t is within the valid ray interval
        if t < t_min or t > t_max:
            return False, None
        
        # Determine the x and y coordinates at this specific t
        x = r.o.x + t * r.d.x
        y = r.o.y + t * r.d.y

        # Boundary check: Does the hit point fall within the rectangle's dimensions?
        if x < self.x0 or x > self.x1 or y < self.y0 or y > self.y1:
            return False, None
        
        # Populate the HitRecord with intersection details
        rec = HitRecord()
        rec.t = t
        rec.p = r.at(t)
        
        # The normal for an XY plane is always along the Z axis (0, 0, 1)
        outward_normal = V3(0, 0, 1)
        rec.set_face_normal(r, outward_normal)
        
        # Calculate UV coordinates for texture mapping (normalized 0.0 to 1.0)
        rec.u = (x - self.x0) / (self.x1 - self.x0)
        rec.v = (y - self.y0) / (self.y1 - self.y0)
        rec.material = self.mat

        return True, rec

    def bounding_box(self, _time0: float, _time1: float) -> tuple[bool, AABB]:
        # AABB must have non-zero volume. Since this is a flat plane, 
        # we add a tiny epsilon (0.0001) to the Z axis so the box isn't infinitely thin.
        return True, AABB(V3(self.x0, self.y0, self.k - 0.0001), V3(self.x1, self.y1, self.k + 0.0001))
    
class XZrect(Hittable):
    """
    Represents a rectangle in the XZ plane at a fixed Y depth (k).
    Commonly used for floors and ceilings.
    """
    def __init__(self, _x0: float, _x1: float, _z0: float, _z1: float, _k: float, _mat: Material) -> None:
        self.x0 = _x0
        self.x1 = _x1
        self.z0 = _z0
        self.z1 = _z1
        self.k = _k # The fixed Y-coordinate
        self.mat = _mat

    def hit(self, r: Ray, t_min: float, t_max: float) -> tuple[bool, HitRecord]:
        # Solve for t on the Y axis
        t = (self.k - r.o.y) / r.d.y
        if t < t_min or t > t_max:
            return False, None
        
        x = r.o.x + t * r.d.x
        z = r.o.z + t * r.d.z

        # Boundary check on X and Z
        if x < self.x0 or x > self.x1 or z < self.z0 or z > self.z1:
            return False, None
        
        rec = HitRecord()
        rec.t = t
        rec.p = r.at(t)
        
        # The normal for an XZ plane is along the Y axis (0, 1, 0)
        outward_normal = V3(0, 1, 0)
        rec.set_face_normal(r, outward_normal)
        
        # UV mapping based on XZ dimensions
        rec.u = (x - self.x0) / (self.x1 - self.x0)
        rec.v = (z - self.z0) / (self.z1 - self.z0)
        rec.material = self.mat

        return True, rec

    def bounding_box(self, _time0: float, _time1: float) -> tuple[bool, AABB]:
        # Padding applied to the Y axis to give the box thickness
        return True, AABB(V3(self.x0, self.k - 0.0001, self.z0), V3(self.x1, self.k + 0.0001, self.z1))
    
class YZrect(Hittable):
    """
    Represents a rectangle in the YZ plane at a fixed X depth (k).
    Commonly used for side walls.
    """
    def __init__(self, _y0: float, _y1: float, _z0: float, _z1: float, _k: float, _mat: Material) -> None:
        self.y0 = _y0
        self.y1 = _y1
        self.z0 = _z0
        self.z1 = _z1
        self.k = _k # The fixed X-coordinate
        self.mat = _mat

    def hit(self, r: Ray, t_min: float, t_may: float) -> tuple[bool, HitRecord]:
        # Solve for t on the X axis
        t = (self.k - r.o.x) / r.d.x
        if t < t_min or t > t_may:
            return False, None
        
        y = r.o.y + t * r.d.y
        z = r.o.z + t * r.d.z

        # Boundary check on Y and Z
        if y < self.y0 or y > self.y1 or z < self.z0 or z > self.z1:
            return False, None
        
        rec = HitRecord()
        rec.t = t
        rec.p = r.at(t)
        
        # The normal for a YZ plane is along the X axis (1, 0, 0)
        outward_normal = V3(1, 0, 0)
        rec.set_face_normal(r, outward_normal)
        
        # UV mapping based on YZ dimensions
        rec.u = (y - self.y0) / (self.y1 - self.y0)
        rec.v = (z - self.z0) / (self.z1 - self.z0)
        rec.material = self.mat

        return True, rec

    def bounding_box(self, _time0: float, _time1: float) -> tuple[bool, AABB]:
        # Padding applied to the X axis to give the box thickness
        return True, AABB(V3(self.k - 0.0001, self.y0, self.z0), V3(self.k + 0.0001, self.y1, self.z1))