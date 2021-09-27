from numpy.linalg.linalg import norm
import pygame
import numpy as np
from numpy import identity as I

white = (255,255,255)
black = (0,0,0)
orange = (255,165,0)


pygame.init()
screenSize = [500, 500]
screen = pygame.display.set_mode((screenSize[0], screenSize[1]))
pygame.display.set_caption('Manifold Test')

exit = False

MAXDEPTH = 5
EPS_F = 0.0000001

# enum
TYPE_PLANE = 0
TYPE_SPHERE = 1

def unit(vec):
    return vec / np.linalg.norm(vec)

def vec3(vec):
    return np.asarray([vec[0], vec[1], 1])

def vec2(vec):
    return np.asarray([vec[0]/vec[2], vec[1]/vec[2]])

class plane:
    def __init__(self, o, n):
        self.o = np.asarray(o)
        self.n = unit(n)

        self.type = TYPE_PLANE

    def endpoints(self):
        s = np.asarray([self.n[1], -self.n[0]])
        t = 999.0
        return [self.o - s * t, self.o + s * t]

    def isect(self, x, v):
        denom = self.n @ v
        if (denom < 0):
            return False
        numer = (self.o - x) @ self.n
        t = numer / denom
        return (t >= 0)

    def isectT(self, x, v):
        denom = self.n @ v
        numer = (self.o - x) @ self.n
        t = numer / denom
        return t

    def reflect(self, v, x):
        return v - 2 * (self.n @ v) * self.n

class sphere:
    def __init__(self, o, r):
        self.o = np.asarray(o)
        self.r = r

        self.type = TYPE_SPHERE

    def endpoints(self):
        return self.o, self.r

    def isect(self, x, v):
        # https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/
        oc = x - self.o
        a = v @ v
        b = 2 * oc @ v
        c = oc @ oc - self.r * self.r
        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return False
        t1 = (-b + np.sqrt(disc)) / (2 * a)
        t2 = (-b - np.sqrt(disc)) / (2 * a)
        if t1 <= 0 and t2 <= 0:
            return False
        return True

    def isectT(self, x, v):
        oc = x - self.o
        a = v @ v
        b = 2 * oc @ v
        c = oc @ oc - self.r * self.r
        disc = b ** 2 - 4 * a * c
        t1 = (-b + np.sqrt(disc)) / (2 * a)
        t2 = (-b - np.sqrt(disc)) / (2 * a)
        if t1 <= 0:
            return t2
        elif t2 <= 0:
            return t1
        return min(t1, t2)

    def n(self, x):
        return unit(x - self.o)

    def reflect(self, v, x):
        n = self.n(x)
        return v - 2 * (n @ v) * n



def set_v0(withMouse=False):
    if withMouse:
        v0 = np.asarray(pygame.mouse.get_pos())
        v0 = v0 - x0
    else:
        v0 = np.random.uniform(-1, 1, 2)
    #v0 = [-1, -0.3]
    return unit(v0)


def set_x0(withMouse=False):
    if withMouse:
        x0 = np.asarray(pygame.mouse.get_pos())
    else:
        x0 = np.random.uniform(0, 1, 2) * screenSize
    #x0 = [400, 400]
    return x0

x = []  # hit points
t = []  # hit times
v = []  # hit velocities
S = []  # scene surfaces

x0 = set_x0()
v0 = set_v0()
t.append(999.0)
S.append(plane([450, 450], [1, 0]))
S.append(plane([450, 450], [0, 1]))
S.append(plane([50, 50], [-1, 0]))
S.append(plane([50, 50], [0, -1]))
#S.append(sphere([50, 50], 50))
#S.append(sphere([500, 250], 80))
#S.append(sphere([50, 400], 150))

s = np.asarray([275, 195])

def forward(x0, v0):
    '''
    input: initial velocity
    output: x list, v list, ss
    '''
    x = []
    v = []
    t = []
    Si = []

    x.append(x0)
    v.append(v0)

    n = 0
    
    hit = True
    while hit:

        if n >= MAXDEPTH:
            break

        hit = False
        hitIdx = 0
        hitT = 0
        for i in range(len(S)):
            if S[i].isect(x[n],v[n]):
                tn = S[i].isectT(x[n], v[n]) - EPS_F
                if not hit:
                    hitIdx = i
                    hitT = tn
                    hit = True
                    
                elif tn < hitT:
                    hitIdx = i
                    hitT = tn

        if hit:
            xn = x[n] + v[n] * hitT
            vn = S[hitIdx].reflect(v[n], xn)
            t.append(hitT)
            x.append(xn)
            v.append(vn)
            Si.append(hitIdx)

            n += 1

    t.append(999.0)

    # compute s*
    ssIdx = -1
    ssMin = 99999999.9
    ssBest = [0,0]
    for i in range(n+1):
        sT = (s - x[i]) @ v[i]
        if sT > 0 and sT < t[i]:
            ss = x[i] + sT * v[i]
            L = np.linalg.norm(s - ss) ** 2
            if L < ssMin:
                ssMin = L
                ssIdx = i
                ssBest = ss
    

    return x, v, t, ssBest, ssIdx, Si

def backward(x, v, t, ss, ssIdx, Si):

    n = ssIdx

    if n == -1:
        return v[0], np.zeros(2)

    # compute loss
    L = np.linalg.norm(s - ss) ** 2

    # compute dL_dv0
    dL_dss = 2 * (ss - s)
    dss_dv = I(2) * ((s - x[n]) @ v[n]) + v[n] * ((s-x[n]) @ I(2))
    #dss_dv = I(2) * ((s - x[n]) @ v[n])
    dL_dv = dL_dss @ dss_dv

    
    if n >= 1:
        for i in range(n, 0, -1):
            surface = S[Si[i-1]]
            dv_dv = I(2)
            if surface.type == TYPE_PLANE:
                normal = surface.n
                dv_dv = I(2) - 2 * (I(2) @ normal) * normal

            if surface.type == TYPE_SPHERE:
                n = surface.n(x[i])
                dv_dv = I(2) - 2 * (I(2) @ normal) * normal
            
            dL_dv = dL_dv @ dv_dv

    

    # optimize
    v[0] -= dL_dv * 0.000001
    v[0] = v[0] / np.linalg.norm(v[0])

    return v[0], dL_dv

LEFT = 1
RIGHT = 3

if True:
    time = 0
    while not exit:

        time += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit = True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT:
            v0 = set_v0(withMouse=True)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == RIGHT:
            x0 = set_x0(withMouse=True)

        if(time % 5000 == 0):
            x, v, t, ss, ssIdx, Si = forward(x0, v0)
            
            screen.fill(white)

            for i in range(len(S)):
                if S[i].type == TYPE_PLANE:
                    line_endpoints = S[i].endpoints()
                    pygame.draw.line(screen, (0,0,0), line_endpoints[0], line_endpoints[1])
                elif S[i].type == TYPE_SPHERE:
                    o, r = S[i].endpoints()
                    pygame.draw.circle(screen, (0,0,0), o, r, 1)

            for i in range(len(x)):
                pygame.draw.circle(screen, (0,0,0), x[i], 5)
                pygame.draw.line(screen, (0,0,0), x[i], x[i] + t[i]*v[i])
            
            pygame.draw.circle(screen, (255,0,0), s, 5)
            pygame.draw.circle(screen, (255,0,0), ss, 5)


            v0, dL_dv = backward(x, v, t, ss, ssIdx, Si)

            '''
            _, _, _, ss, _, _ = forward(x0, v0)
            L00 = np.linalg.norm(s - ss) ** 2
            _, _, _, ss, _, _ = forward(x0, v0 + np.asarray([EPS_F,0]))
            L10 = np.linalg.norm(s - ss) ** 2
            _, _, _, ss, _, _ = forward(x0, v0 + np.asarray([0,EPS_F]))
            L01 = np.linalg.norm(s - ss) ** 2
            dL_dvx = (L10 - L00) / EPS_F
            dL_dvy = (L01 - L00) / EPS_F


            x, v, t, ss, ssIdx, Si = forward(x0, v0)

            v0, dL_dv = backward(x, v, t, ss, ssIdx, Si)

            dL_dv_num = -np.asarray([dL_dvx, dL_dvy])
            dL_dv_ana = -dL_dv

            #v[0] += dL_dv_num * 0.000001
            #v[0] = v[0] / np.linalg.norm(v[0])

            s_test = x0 + v0 * t[0] + (v0 - 2 * (S[Si[0]].n @ v0) * S[Si[0]].n) * ((s - x[1]) @ v[1])

            if np.isnan(np.sum(dL_dv_num)):
                dL_dv_num = np.zeros(2)

            if np.isnan(np.sum(dL_dv_ana)):
                dL_dv_ana = np.zeros(2)
            
            pygame.draw.line(screen, (0,255,0), x0, x0 + dL_dv_num/100)
            pygame.draw.line(screen, (0,0,255), x0, x0 + dL_dv_ana/100)

            pygame.draw.circle(screen, (255,255,0), s_test, 5)
            '''
            pygame.display.update()

pygame.quit()
quit()