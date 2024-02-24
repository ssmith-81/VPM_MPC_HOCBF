# panel_functions.py

import numpy as np

# Find geometric quantities of the obstacle
def CLOVER_COMPONENTS(xa, ya, U_inf, V_inf, g_source, g_sink, xs, ys, xsi, ysi, n, g_clover, x_cl, y_cl):
    xmid = np.zeros(n)
    ymid = np.zeros(n)
    dx = np.zeros(n)
    dy = np.zeros(n)
    Sj = np.zeros(n)
    phiD = np.zeros(n)
    rhs = np.zeros(n+1)  # Define the extra point for kutta condition. We will place this value in CLOVER_KUTTA function
    # So when we define rhs[n]=... in the CLOVER_KUTTA function, that is where it is defined!

    for i in range(n):
        xmid[i] = (xa[i] + xa[i + 1]) / 2
        ymid[i] = (ya[i] + ya[i + 1]) / 2
        dx[i] = xa[i + 1] - xa[i]
        dy[i] = ya[i + 1] - ya[i]
        Sj[i] = np.sqrt(dx[i] ** 2 + dy[i] ** 2)
        phiD[i] = np.degrees(np.arctan2(dy[i], dx[i]))

        if phiD[i] < 0:
            phiD[i] += 360

        rhs[i] = U_inf * ymid[i] - V_inf * xmid[i] \
                 + (g_source / (2 * np.pi)) * np.arctan2(ymid[i] - ys, xmid[i] - xs) \
                 - (g_sink / (2 * np.pi)) * np.arctan2(ymid[i] - ysi, xmid[i] - xsi) + (g_clover / (2*np.pi))*np.arctan2(ymid[i] - y_cl, xmid[i] - x_cl)

    return xmid, ymid, dx, dy, Sj, phiD, rhs

def CLOVER_STREAM_GEOMETRIC_INTEGRAL(xmid, ymid, xa, ya, phi, Sj, n):
    I = np.zeros((n+1, n+1)) # Remember, the rows and columns start at position (0,0).
    # so technically matlab n = python n-1 and matlab n+1 = python n
    # The n+1 places are the boundary condition at panel j=n+1 and Kutta condition at point 
    # i=n+1. We will place values there later one. But remember, we do it by I(n,n)=... because 
    # np.zeros(n+1,n+1) sets rows and columns as 0->n

    for i in range(n):
        for j in range(n+1): # from 0->n (matlab 1->n+1)
            if j == n: # would be n+1 in matlab
                I[i, j] = 1  # Equation (16)
            else:
                # Transform the current control point i into the current panel j reference frame
                x_o = xmid[i] - xa[j]
                y_o = ymid[i] - ya[j]
                phi_j = phi[j]

                # Rotate about origin o to complete the transformation
                xbar = x_o * np.cos(phi_j) + y_o * np.sin(phi_j)
                ybar = -x_o * np.sin(phi_j) + y_o * np.cos(phi_j)

                # Calculate r1 and r2 between the current panel j endpoints and the current control point i
                r1 = np.sqrt(xbar**2 + ybar**2)
                r2 = np.sqrt((xbar - Sj[j])**2 + ybar**2)

                # Calculate omega angles between each control point i and each panel endpoints j
                omega1 = np.arctan2(ybar, xbar)
                omega2 = np.arctan2(ybar, xbar - Sj[j])

                # Compute I_(i,j) geometric values for each panel j on control point i
                I[i, j] = -(1 / (2 * np.pi)) * (xbar * np.log(r2 / r1) - Sj[j] * np.log(r2) + ybar * (omega1 - omega2))

    return I

def CLOVER_KUTTA(I, trail_point, xa, ya, phi, Sj, n, flagKutta, rhs, U_inf, V_inf, xs, ys, xsi, ysi, g_source, g_sink, g_clover, x_cl, y_cl):
    # Form the last line of equation with the Kutta condition
    if flagKutta[0] == 1:
        I[n, 0] = 1
        I[n, n-1] = 1 
        I[n, n] = 0 
        rhs[n] = 0
        for j in range(1, n-1):
            I[n, j] = 0

    # This is for a sail like object i.e. lidar detection (non closed object)
    # Maybe have a condition that alters between gamme_1 = 0 and gamme_N = 0 depending on whether the obstacle is left or right of center relative to the Clover.
    if flagKutta[2] == 1: # gamma_N = 0 -> slow smoothly off end of sail
        I[n, 0] = 1
        I[n, n-1] = 0  # gamma_N = rhs[n] = 0 
        I[n, n] = 0  # stream_function/psi position
        rhs[n] = 0
        for j in range(1, n-1):
            I[n, j] = 0

# This is for an extended trail point for a non-closed sail like object
    if flagKutta[3] == 1:
            rhs[n] = trail_point[1] * U_inf - trail_point[0] * V_inf + (g_source / (2 * np.pi)) * np.arctan2(
                trail_point[1] - ys, trail_point[0] - xs) - (g_sink / (2 * np.pi)) * np.arctan2(
                trail_point[1] - ysi, trail_point[0] - xsi) + (g_clover / (2*np.pi))*np.arctan2(trail_point[1] - y_cl, trail_point[0] - x_cl) # add random source here to see what happens

            for j in range(n+1):
                if j == n:
                    I[n, j] = 1  # Equation (16)
                else:
                    x_o = trail_point[0] - xa[j]
                    y_o = trail_point[1] - ya[j]

                    # Rotate about origin o to complete the transformation
                    xbar = x_o * np.cos(phi[j]) + y_o * np.sin(phi[j])
                    ybar = -x_o * np.sin(phi[j]) + y_o * np.cos(phi[j])

                    # Calculate r1 and r2 between the current panel j endpoints and the current control point i
                    r1 = np.sqrt(xbar**2 + ybar**2)
                    r2 = np.sqrt((xbar - Sj[j])**2 + ybar**2)

                    # Calculate omega angles between each control point i
                    # and each panel endpoints j (store for calculating normal and tangential velocities later):
                    omega1 = np.arctan2(ybar, xbar)
                    omega2 = np.arctan2(ybar, xbar - Sj[j])

                    # Compute I_(i,j) geometric values for each panel j on control point i
                    I[n, j] = -(1 / (2 * np.pi)) * (xbar * np.log(r2 / r1) - Sj[j] * np.log(r2) + ybar * (omega1 - omega2))

    return I, rhs

def CLOVER_STREAMLINE(xmid, ymid, xa, ya, phi, g, Sj, U_inf, V_inf, xs, ys, xsi, ysi, g_source, g_sink, g_clover, x_cl, y_cl):
    # Number of panels
    n = len(xa) - 1

    u = 0
    v = 0

    # Compute vortex velocity contributions
    for j in range(n):
        # Transform the current control point i into the current panel j reference frame (translation):
        x_o = xmid - xa[j]
        y_o = ymid - ya[j]

        # Rotate about origin o to complete transformation
        xbar = x_o * np.cos(phi[j]) + y_o * np.sin(phi[j])
        ybar = -x_o * np.sin(phi[j]) + y_o * np.cos(phi[j])

        # First calculate r1 and r2
        r1 = np.sqrt(xbar**2 + ybar**2)
        r2 = np.sqrt((xbar - Sj[j])**2 + ybar**2)

        # Calculate omega angles:
        omega1 = np.arctan2(ybar, xbar)
        omega2 = np.arctan2(ybar, xbar - Sj[j])

        # Calculate the vortex velocity contributions in the panel j reference frame:
        u_panel = (g[j] / (2 * np.pi)) * (omega1 - omega2)
        v_panel = -(g[j] / (2 * np.pi)) * np.log(r2 / r1)

        # Transform velocity contributions back to the global frame:
        u += u_panel * np.cos(phi[j]) - v_panel * np.sin(phi[j])
        v += u_panel * np.sin(phi[j]) + v_panel * np.cos(phi[j])

    # Calculate the source and sink contributions
    u_source = (g_source / (2 * np.pi)) * ((xmid - xs) / ((xmid - xs)**2 + (ymid - ys)**2))
    u_sink = -(g_sink / (2 * np.pi)) * ((xmid - xsi) / ((xmid - xsi)**2 + (ymid - ysi)**2))

    v_source = (g_source / (2 * np.pi)) * ((ymid - ys) / ((xmid - xs)**2 + (ymid - ys)**2))
    v_sink = -(g_sink / (2 * np.pi)) * ((ymid - ysi) / ((xmid - xsi)**2 + (ymid - ysi)**2))

     # introduce source on clover drone
    
    u_clover = (g_clover / (2 * np.pi)) * ((xmid - x_cl) / ((xmid - x_cl)**2 + (ymid - y_cl)**2))
    v_clover = (g_clover / (2 * np.pi)) * ((ymid - y_cl) / ((xmid - x_cl)**2 + (ymid - y_cl)**2))

    # Include the uniform flow contributions to the velocity calculations:
    u += U_inf + u_source + u_sink + u_clover
    v += V_inf + v_source + v_sink + v_clover

    return u, v

def CLOVER_noOBSTACLE(xmid, ymid, U_inf, V_inf, xs, ys, xsi, ysi, g_source, g_sink, g_clover, x_cl, y_cl):
   

    u = 0
    v = 0
   

    # Calculate the source and sink contributions
    u_source = (g_source / (2 * np.pi)) * ((xmid - xs) / ((xmid - xs)**2 + (ymid - ys)**2))
    u_sink = -(g_sink / (2 * np.pi)) * ((xmid - xsi) / ((xmid - xsi)**2 + (ymid - ysi)**2))

    v_source = (g_source / (2 * np.pi)) * ((ymid - ys) / ((xmid - xs)**2 + (ymid - ys)**2))
    v_sink = -(g_sink / (2 * np.pi)) * ((ymid - ysi) / ((xmid - xsi)**2 + (ymid - ysi)**2))

     # introduce source on clover drone
    
    u_clover = (g_clover / (2 * np.pi)) * ((xmid - x_cl) / ((xmid - x_cl)**2 + (ymid - y_cl)**2))
    v_clover = (g_clover / (2 * np.pi)) * ((ymid - y_cl) / ((xmid - x_cl)**2 + (ymid - y_cl)**2))

    # Include the uniform flow contributions to the velocity calculations:
    u += U_inf + u_source + u_sink + u_clover
    v += V_inf + v_source + v_sink + v_clover

    return u, v

