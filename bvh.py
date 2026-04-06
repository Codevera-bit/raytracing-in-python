from random import randint
from hittable import *
from hittablelist import *
from functools import cmp_to_key

class BVHnode(Hittable):
    """
    A node in the Bounding Volume Hierarchy tree. 
    Each node represents a volume of space containing a subset of the scene's objects.
    """
    def __init__(self, lst: HittableList, start: int, end: int, time0: float, time1: float) -> None:
        # Randomly choose an axis (x, y, or z) to split the objects.
        # Splitting on a random axis helps keep the tree balanced on average.
        axis = randint(0, 2)

        if axis == 0:
            comparator = box_x_compare
        elif axis == 1:
            comparator = box_y_compare
        else:
            comparator = box_z_compare

        object_span = end - start

        if object_span == 1:
            # Base Case 1: Only one object. 
            # We duplicate it to keep the binary tree structure consistent.
            self.left = lst.objects[start]
            self.right = lst.objects[start]
            
        elif object_span == 2:
            # Base Case 2: Two objects. 
            # We simply sort them and assign one to left, one to right.
            if comparator(lst.objects[start], lst.objects[start + 1]):
                self.left = lst.objects[start]
                self.right = lst.objects[start + 1]
            else:
                self.left = lst.objects[start + 1]
                self.right = lst.objects[start]
                
        else:
            # Recursive Step: Sort the objects along the chosen axis 
            # and split the list in half to create child nodes.
            lst.objects.sort(key=cmp_to_key(comparator))
            
            mid = start + object_span // 2
            self.left = BVHnode(lst, start, mid, time0, time1)
            self.right = BVHnode(lst, mid, end, time0, time1)

        # Calculate the bounding box for this node by merging the boxes of its children.
        b1, box_left = self.left.bounding_box(time0, time1)
        b2, box_right = self.right.bounding_box(time0, time1)

        if not (b1 and b2):
            print("ERROR: No bounding box in BVHNode __init__.")

        self.box = surrounding_box(box_left, box_right)
    
    def hit(self, r: Ray, t_min: float, t_max: float) -> tuple[bool, HitRecord]:
        """
        The core acceleration logic: 
        If the ray doesn't hit this node's box, it can't hit anything inside it.
        """
        # 1. Check the bounding box of this entire branch.
        if not self.box.hit(r, t_min, t_max):
            return False, None

        # 2. Check the left child.
        hit_left, rec_left = self.left.hit(r, t_min, t_max)
        
        # 3. Check the right child. 
        # Note: If we hit the left side, we shrink t_max to 'rec_left.t'.
        # This ensures we only care about hits on the right that are CLOSER than the left hit.
        if hit_left:
            hit_right, rec_right = self.right.hit(r, t_min, rec_left.t)
        else:
            hit_right, rec_right = self.right.hit(r, t_min, t_max)

        # Return the closest record found.
        if hit_right:
            return True, rec_right
        if hit_left:
            return True, rec_left
            
        return False, None

    def bounding_box(self, _time0: float, _time1: float) -> tuple[bool, AABB]:
        """Returns the pre-calculated bounding box for this node."""
        return True, self.box
    
# --- Helper Functions for Sorting ---

def box_compare(a: Hittable, b: Hittable, axis: int) -> bool:
    """
    Generic comparator that checks which of two objects starts earlier 
    along a specific axis (0=x, 1=y, 2=z).
    """
    b1, box_a = a.bounding_box(0, 0)
    b2, box_b = b.bounding_box(0, 0)

    if not b1 or not b2:
        print("ERROR: No bounding box in BVHNode __init__.")

    # Access the minimum coordinate for the chosen axis.
    if axis == 0:
        return box_a.mini.x < box_b.mini.x
    if axis == 1:
        return box_a.mini.y < box_b.mini.y
    
    return box_a.mini.z < box_b.mini.z

# Specific wrappers for the sorting functions
def box_x_compare(a: Hittable, b: Hittable) -> bool:
    return box_compare(a, b, 0)

def box_y_compare(a: Hittable, b: Hittable) -> bool:
    return box_compare(a, b, 1)

def box_z_compare(a: Hittable, b: Hittable) -> bool:
    return box_compare(a, b, 2)