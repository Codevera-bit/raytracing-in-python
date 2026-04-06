from functools import cmp_to_key
from hittable import *

class HittableList(Hittable):
    """
    A collection of hittable objects. This class allows you to treat 
    an entire scene (or a group of objects) as a single object.
    """
    def __init__(self) -> None:
        # Stores all objects (spheres, boxes, etc.) that make up the scene
        self.objects = []

    def sort(self, cmp: 'function') -> None:
        """Sorts the objects in the list using a custom comparator function."""
        self.objects = sorted(self.objects, key=cmp_to_key(cmp))

    def clear(self) -> None:
        """Removes all objects from the list."""
        self.objects.clear()

    def add(self, obj: Hittable) -> None:
        """Adds a new object to the scene."""
        self.objects.append(obj)

    def hit(self, r: Ray, t_min: float, t_max: float) -> tuple[bool, HitRecord]:
        """
        Calculates the closest intersection point between a ray and any 
        object in the list.
        """
        rec = HitRecord()
        hit_anything = False
        # As we find hits, we shrink 'closest_so_far' so we don't 
        # waste time checking objects further away than what we've already hit.
        closest_so_far = t_max

        for obj in self.objects:
            has_hit, temp_rec = obj.hit(r, t_min, closest_so_far)
            if has_hit:
                hit_anything = True
                # Update the horizon: only hits closer than this one now matter
                closest_so_far = temp_rec.t

                # Manually copy all hit data from the temporary record to our main record
                rec.p = temp_rec.p
                rec.normal = temp_rec.normal
                rec.t = temp_rec.t
                rec.u = temp_rec.u
                rec.v = temp_rec.v
                rec.front_face = temp_rec.front_face
                rec.material = temp_rec.material

        return hit_anything, rec

    def bounding_box(self, _time0: float, _time1: float) -> tuple[bool, AABB]:
        """
        Computes a single AABB that encloses every object in the list.
        Returns (True, box) if successful, (False, None) if the list is empty 
        or contains an unbounded object.
        """
        if len(self.objects) == 0:
            return False, None
        
        output_box = None
        first_box = True

        for obj in self.objects:
            # Check if the individual object has a bounding box
            b, temp_box = obj.bounding_box(_time0, _time1)
            if not b:
                return False, None # If one object can't be bounded, the whole list can't
            
            if first_box:
                # The first valid box found becomes our starting volume
                output_box = temp_box
                first_box = False
            else:
                # Expand the output_box to include the current object's box
                output_box = surrounding_box(output_box, temp_box)

        # IMPORTANT: Return both the success flag and the calculated box
        return True, output_box