import numpy as np
def Musk(inflow, k, x, dt):
    c0 = (-k*x+0.5*dt)/(k-k*x+0.5*dt)
    c1 = (k*x+0.5*dt)/(k-k*x+0.5*dt)
    c2 = (k-k*x-0.5*dt)/(k-k*x+0.5*dt)
    outflow = np.zeros_like(inflow)
    outflow[0] = inflow[0]
    for i in range(1, len(inflow)):
        outflow[i] = c0*inflow[i] + c1*inflow[i-1] + c2*outflow[i-1]
    return outflow
