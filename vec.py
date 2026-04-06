import math
import random
import numpy as np

class V3:
    def __init__(self, x, y=None, z=None):
        if y is None and z is None:
            if hasattr(x, '__iter__') and not isinstance(x, str):
                self.data = np.array(x, dtype=float)
            else:
                self.data = np.array([x, x, x], dtype=float)
        else:
            self.data = np.array([x, y, z], dtype=float)

    @property
    def x(self):
        return self.data[0]

    @x.setter
    def x(self, value):
        self.data[0] = value

    @property
    def y(self):
        return self.data[1]

    @y.setter
    def y(self, value):
        self.data[1] = value

    @property
    def z(self):
        return self.data[2]

    @z.setter
    def z(self, value):
        self.data[2] = value

    def __getitem__(self, idx):
        return self.data[idx]

    def neg(self):
        return V3(-self.data)

    def len_sqr(self):
        return np.sum(self.data ** 2)

    def len(self):
        return np.sqrt(self.len_sqr())

    def near_zero(self):
        s = 1e-8
        return np.all(np.abs(self.data) < s)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

def vec_add(u: V3, v: V3) -> V3:
    return V3(u.data + v.data)

def vec_sub(u: V3, v: V3) -> V3:
    return V3(u.data - v.data)

def vec_mul(u: V3, v: V3) -> V3:
    return V3(u.data * v.data)

def vec_smul(v: V3, t: float) -> V3:
    return V3(t * v.data)

def vec_sdiv(v: V3, t: float) -> V3:
    return V3(v.data / t)

def vec_dot(u: V3, v: V3) -> float:
    return np.sum(u.data * v.data)

def vec_cross(u: V3, v: V3) -> V3:
    return V3(np.cross(u.data, v.data))

def vec_unit(v: V3) -> V3:
    return V3(v.data / v.len())

def vec_rand() -> V3:
    return V3(np.random.random(3))

def vec_rand_between(min: float, max: float) -> V3:
    return V3(np.random.uniform(min, max, 3))

def vec_rand_in_unit_sphere() -> V3:
    while True:
        p = vec_rand_between(-1, 1)
        if p.len_sqr() >= 1:
            continue
        return p

def vec_rand_unit_in_unit_sphere() -> V3:
    return vec_unit(vec_rand_in_unit_sphere())

def vec_rand_in_unit_disk() -> V3:
    while True:
        p = V3(np.random.uniform(-1, 1, 3))
        p.data[2] = 0
        if p.len_sqr() >= 1:
            continue
        return p

def vec_reflect(v: V3, n: V3) -> V3:
    return vec_sub(v, vec_smul(n, 2 * vec_dot(v, n)))

def vec_refract(uv: V3, n: V3, etai_over_etat: float) -> V3:
    cos_theta = min(vec_dot(uv.neg(), n), 1.0)
    r_out_perp = vec_smul(vec_add(uv, vec_smul(n, cos_theta)), etai_over_etat)
    r_out_parallel = vec_smul(n, -math.sqrt(math.fabs(1.0 - r_out_perp.len_sqr())))

    return vec_add(r_out_perp, r_out_parallel)
