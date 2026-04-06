from vec import *
from ray import *

class AABB:
    """
    Axis-Aligned Bounding Box (AABB).
    Used for volume-based collision detection and ray-tracing acceleration structures 
    like BVHs (Bounding Volume Hierarchies).
    """
    def __init__(self, mini: V3, maxi: V3) -> None:
        """
        Initializes the box using two points representing the 'minimum' and 'maximum' 
        extents (e.g., bottom-left-front and top-right-back).
        """
        self.mini = mini
        self.maxi = maxi

    def hit(self, r: Ray, t_min: float, t_max: float) -> bool:
        """
        Determines if a ray intersects the box within a specific interval [t_min, t_max].
        Uses the Slab Method: the box is the intersection of three pairs of parallel planes.
        """
        for a in range(3):
            # Calculate the inverse direction to avoid repeated division and 
            # handle cases where the ray is parallel to an axis.
            invd = 1.0 / r.d[a]
            
            # Compute the distance from the ray origin to the 'near' and 'far' 
            # planes for the current axis (x, y, or z).
            t0 = (self.mini[a] - r.o[a]) * invd
            t1 = (self.maxi[a] - r.o[a]) * invd

            # If the ray direction is negative, t0 will be greater than t1.
            # We swap them so t0 always represents the entry and t1 the exit.
            if invd < 0.0:
                t0, t1 = t1, t0

            # Narrow the interval:
            # The ray must enter ALL three slabs before it exits ANY of them.
            if t0 > t_min:
                t_min = t0  # The latest entry point
            
            if t1 < t_max:
                t_max = t1  # The earliest exit point

            # If the exit point is before the entry point, the ray missed the box.
            if t_max <= t_min:
                return False
        
        # If we survive all three axes, the ray passes through the box.
        return True
    
def surrounding_box(box0: AABB, box1: AABB) -> AABB:
    """
    Computes a new AABB that perfectly encloses two existing AABBs.
    This is essential for building tree structures (like BVHs) where 
    parent nodes must bound all their children.
    """
    # Take the minimum coordinates of both boxes to find the new overall 'mini'
    small = V3(
        min(box0.mini.x, box1.mini.x),
        min(box0.mini.y, box1.mini.y),
        min(box0.mini.z, box1.mini.z)
    )

    # Take the maximum coordinates of both boxes to find the new overall 'maxi'
    big = V3(
        max(box0.maxi.x, box1.maxi.x),
        max(box0.maxi.y, box1.maxi.y),
        max(box0.maxi.z, box1.maxi.z)
    )

    return AABB(small, big)