import numpy as np

def quatMul(q0, q1):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                      x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    
def quatConj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)
def quatNorm2(q):
    w, x, y, z = q
    return (w**2 + x**2 + y**2 + z**2)
def quatNorm(q):
    return np.sqrt(quatNorm2(q))
def quatNormalize(q):
    w, x, y, z = q
    n = quatNorm(q)
    return np.array([w/n, x/n, y/n, z/n], dtype=np.float64)
def quatInv(q):
    return quatConj(q) / quatNorm2(q)

def quatFromAngleAxis(angle, axis):
    u,v,r = axis
    cos = np.cos(angle/2.)
    sin = np.sin(angle/2.)
    return quatNormalize(np.array([cos, sin*u, sin*v, sin*r], dtype=np.float64))
def pureQuat(vector):
    return np.concatenate([[0], vector])
def getPure(q):
    w, x, y, z = q
    return np.array([x, y, z], dtype=np.float64)

def quatTrans(q,v):
    """
    Rotation a vector using a quaternion.
    """
    return getPure(quatMul(q, quatMul(pureQuat(v), quatConj(q))))


def quat_dot_np(p, q):
    return np.sum(p*q)
def quat_dist_np(p, q):
    # See: https://math.stackexchange.com/a/90098
    # d(p,q) = 1 - <p,q>^2
    return 1 - (quat_dot_np(p, q))**2


if __name__=="__main__":
    ## Test quaternions
    q = quatFromAngleAxis(np.pi/2, [0,1,0])
    print(q)
    print("[0.70710678, 0., 0.70710678, 0.]")
    # quatMul(q, quatInv(q))
    # quatFromAngleAxis(np.pi/2, [1,0,0])
    print(getPure(quatMul(q,quatMul(pureQuat([1,2,3]), quatConj(q)))))
    print("array([-3.,  2.,  1.])")
