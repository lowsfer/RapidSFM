#!/usr/bin/env python3
from sympy import *

g0, g1, g2, t0, t1 = symbols('g0, g1, g2, t0, t1', real=True, commutative=True)
g = Matrix([g0, g1, g2])

def gvec2mat(g):
    r = Matrix.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            r[j, i] = ((1 - g.dot(g)) * KroneckerDelta(i, j) + 2 * g[i] * g[j] + 2 * sum([LeviCivita(i, j, k) * g[k] for k in range(3)])) / (1 + g.dot(g))
    return r

def T2Tgvec(T):
    length = sqrt(T[0]**2 + T[1]**2 + T[2]**2)
    #this implementation have problem when T[0]**2 + T[1]**2 approaches zero
    #sin_theta = sqrt(T[0]**2 + T[1]**2) / length
    #cos_theta = T[2] / length
    #tan_half_theta = (1 - cos_theta) / sin_theta
    #return Matrix([T[1], -T[0], 0]) / sqrt(T[0]**2 + T[1]**2) * tan_half_theta
    #above can be refactored to below:
    return simplify(1/(length + T[2]) * Matrix([-T[1],T[0], 0]))


t_gvec = Matrix([t0, t1, 0])

def gvec_mul(b, a):# f*g, i.e. apply a first, then apply b rotation, equivalent to gvec2mat(b) * gvec2mat(a)
    return (a + b + b.cross(a)) / (1 - a.dot(b))

#T0, T1, T2 = symbols('T0, T1, T2', real=True, commutative=True)
#T = gvec2mat(T2Tgvec(Matrix([T0, T1, T2]))) * gvec2mat(t_gvec) * Matrix([0,0,1])


TR00, TR01, TR02, TR10, TR11, TR12, TR20, TR21, TR22 = symbols('TR00, TR01, TR02, TR10, TR11, TR12, TR20, TR21, TR22', real=True, commutative=True)

T = Matrix([TR00, TR01, TR02, TR10, TR11, TR12, TR20, TR21, TR22]).reshape(3,3) * gvec2mat(t_gvec) * Matrix([0,0,1])


R00, R01, R02, R10, R11, R12, R20, R21, R22 = symbols('R00, R01, R02, R10, R11, R12, R20, R21, R22', real=True, commutative=True)
r = gvec2mat(g)
R = Matrix([R00, R01, R02, R10, R11, R12, R20, R21, R22]).reshape(3,3) * r


RT = Matrix.hstack(R,T)
fx,fy,cx,cy = symbols('fx,fy,cx,cy', real=True, commutative=True)
K = Matrix([[fx,0,cx],[0,fy,cy],[0,0,1]])
px,py,pz = symbols('px,py,pz', real=True, commutative=True)
X2h = K*RT*Matrix([px,py,pz,1])
X2 = Matrix([X2h[0]/X2h[2], X2h[1]/X2h[2]])
