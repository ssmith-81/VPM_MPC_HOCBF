import rospy
import json
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from CLOVER_MODEL import export_clover_model
from FUNCTIONS import CloverCost, acados_settings
import numpy as np
import math
import scipy.linalg
from casadi import SX, vertcat, sin, cos, norm_2, diag, sqrt 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Global lof variables
X = []
VX = []
Y = []
U = []
Z = []
VZ = []
#Obstacle dynamics
x_o = []
y_o = []
z_o = []
vx_o = []
vy_o = []
vz_o = []
ax_o = []
ay_o = []
az_o = []




# This class categorizes all of the functions used for complex rajectory tracking
class clover:

    def __init__(self, N_horizon, T_horizon, Time): 
        self.X0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Initialize the states [x,xdot,y,ydot,z,zdot]
        self.N_horizon = N_horizon # Define prediction horizone in terms of optimization intervals
        self.T_horizon = T_horizon # Define the prediction horizon in terms of time (s) --> Limits time and improves efficiency
        # I think we will have to set this to 1/50 when using for position controller in PX4? Because we need the overall
        # constroller operating at 50 Hz
        self.Time = Time  # maximum simulation time[s]

        Nsim = int(self.Time * self.N_horizon / self.T_horizon)

        i    = 0                        # Set the counter
        dt   = self.T_horizon/self.N_horizon		# Set the sample time step
        dadt = math.pi*2 / self.Time # first derivative of angle with respect to time
        r    = 4		# Set the radius of the figure-8

        # initialize the obstacles trajectory
        # create random time array with enough elements to complete the entire figure-8 sequence
        t = np.arange(0,Nsim,1)
		
        # Create arrays for each variable we want to feed information to:
        self.posx = [1]*len(t)
        self.posy = [1]*len(t)
        self.posz = [1]*len(t)
        self.velx = [1]*len(t)
        self.vely = [1]*len(t)
        self.velz = [1]*len(t)
        self.afx = [1]*len(t)
        self.afy = [1]*len(t)
        self.afz = [1]*len(t)
        
		
        for i in range(0, Nsim):
		
			# calculate the parameter 'a' which is an angle sweeping from -pi/2 to 3pi/2
			# through the figure-8 curve. 
            a = (-math.pi/2) + i*(math.pi*2/Nsim)
            # These are definitions that will make position, velocity, and acceleration calulations easier:
            c = math.cos(a)
            c2a = math.cos(2.0*a)
            c4a = math.cos(4.0*a)
            c2am3 = c2a-3.0
            c2am3_cubed = c2am3*c2am3*c2am3
            s = math.sin(a)
            cc = c*c
            ss = s*s
            sspo = (s*s)+1.0 # sin squared plus one
            ssmo = (s*s)-1.0 # sin squared minus one
            sspos = sspo*sspo

            # For more information on these equations, refer to the GitBook Clover documentation:

            # Position
            # https:#www.wolframalpha.com/input/?i=%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29
            self.posx[i] = -(r*c*s) / sspo + 15 # 2.5
            # https:#www.wolframalpha.com/input/?i=%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29
            self.posy[i] =  (r*c)   / sspo + 15 # 3.5
            self.posz[i] =  0

            # Velocity
            # https:#www.wolframalpha.com/input/?i=derivative+of+%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
            self.velx[i] =   dadt*r* ( ss*ss + ss + (ssmo*cc) ) / sspos
            # https:#www.wolframalpha.com/input/?i=derivative+of+%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
            self.vely[i] =  -dadt*r* s*( ss + 2.0*cc + 1.0 )    / sspos
            self.velz[i] =  0.0

            # Acceleration
            # https:#www.wolframalpha.com/input/?i=second+derivative+of+%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
            self.afx[i] =  -dadt*dadt*8.0*r*s*c*((3.0*c2a) + 7.0)/ c2am3_cubed
            # https:#www.wolframalpha.com/input/?i=second+derivative+of+%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
            self.afy[i] =  dadt*dadt*r*c*((44.0*c2a) + c4a - 21.0) / c2am3_cubed
            self.afz[i] =  0.0

	



    def main(self):

        # load model
        acados_solver, acados_integrator, model = acados_settings(self.N_horizon, self.T_horizon)


        # prepare simulation Matlab t = 0:1/50:10 -> length = 1:501
        
        # dimensions
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        Nsim = int(self.Time * self.N_horizon / self.T_horizon)

        # Arrays to store the results (chatgpt)
        # simX = np.ndarray((Nsim, nx))  # xHistory   or np.ndarray((Nsim+1,nx))??
        # simU = np.ndarray((Nsim - 1, nu))     # uHistory  or np.ndarray((Nsim, nu))??
        simX = np.ndarray((Nsim + 1, nx))  # xHistory   or np.ndarray((Nsim+1,nx))??
        simU = np.ndarray((Nsim, nu))     # uHistory  or np.ndarray((Nsim, nu))??

        # initialize data structs
        # simX = np.ndarray((Nsim, nx))
        # simU = np.ndarray((Nsim, nu))

        timings = np.zeros((Nsim,))

        x_current = self.X0
        simX[0,:] = x_current # Set the initial state

        # Set the reference trajectory for the cost function
        #self.ocp = self.generate_ref(self.N_sim)

        # # initialize solver
        # for stage in range(N_horizon + 1):
        #     acados_solver.set(stage, "x", 0.0 * np.ones(x_current.shape))
        # for stage in range(N_horizon):
        #     acados_solver.set(stage, "u", np.zeros((nu,)))

        # Closed loop simulation

        k=0

        for i in range(Nsim):

            # # set initial state constraint
            # acados_solver.set(0, 'lbx', x_current) # The 0 represents the first shooting node, or initial position
            # acados_solver.set(0, 'ubx', x_current)

            # update reference
            for j in range(self.N_horizon):
               # yref = np.array([s0 + (sref - s0) * j / N, 0, 0, 0, 0, 0, 0, 0])
                #yref=np.array([1,0,1,0,1,0,0,0,0]) # Set a constant reference of 1 for each position for now
                yref=np.array([0,0.5,0,0.5,0,0.2,0,0,0]) # Constant velocity
                acados_solver.set(j, "yref", yref)
                # acados_solver.set(j, "p", np.array([8,0,0,6,0,0])) # set the obstacle dynamics [x,vx,ax,y,vy,ay] assume static circle
                index = k+j
                if index < len(self.posx):
                    acados_solver.set(j, "p", np.array([self.posx[index],self.velx[index],self.afx[index],self.posy[index],self.vely[index],self.afy[index]])) # set the obstacle dynamics [x,vx,ax,y,vy,ay] dynamic obstacle
                else:
                    acados_solver.set(j, "p", np.array([self.posx[-1],self.velx[-1],self.afx[-1],self.posy[-1],self.vely[-1],self.afy[-1]]))

            #yref_N = np.array([1,0,1,0,1,0]) # Terminal position reference
            yref_N = np.array([0,0.5,0,0.5,0,0.2]) # terminal velocity constraint
            # yref_N=np.array([0,0,0,0,0,0])
            acados_solver.set(self.N_horizon, "yref", yref_N)
            # acados_solver.set(self.N_horizon, "p", np.array([8,0,0,6,0,0])) # State = [x, vx, ax, y, vy, ay]
            index2 = k + self.N_horizon
            if index2 < len(self.posx):
                acados_solver.set(self.N_horizon, "p", np.array([self.posx[index2],self.velx[index2],self.afx[index2],self.posy[index2],self.vely[index2],self.afy[index2]])) # State = [x, vx, ax, y, vy, ay]
            else:
                acados_solver.set(self.N_horizon, "p", np.array([self.posx[-1],self.velx[-1],self.afx[-1],self.posy[-1],self.vely[-1],self.afy[-1]]))

            # Solve ocp
            status = acados_solver.solve()
            # timings[i] = acados_solver.get_status("time_tot")

            if status not in [0, 2]:
                acados_solver.print_statistics()
                raise Exception(
                    f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
                )

            if status == 2:
                print(
                    f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
                )

            # if status != 0:
            #     raise Exception('acados acados_ocp_solver returned status {} in time step {}. Exiting.'.format(status, i))

            # get solution
            x0 = acados_solver.get(0, "x")
            u0 = acados_solver.get(0, "u")
        
            print("control at time", i, ":", simU[i,:])
            for j in range(nx):
                simX[i, j] = x0[j]
            for j in range(nu):
                simU[i, j] = u0[j]
            # simU[i,:] = acados_solver.get(0, "u")
            

            # simulate system
            # acados_integrator.set("x", x_current)
            # acados_integrator.set("u", simU[i,:])

            # status = acados_integrator.solve()
            # if status != 0:
            #     raise Exception(
            #         f"acados integrator returned status {status} in closed loop instance {i}"
            #     )

            # update initial condition
            x0 = acados_solver.get(1, "x")
            acados_solver.set(0, "lbx", x0) # Update the zero shooting node position
            acados_solver.set(0, "ubx", x0)
            
			# logging/debugging
            X.append(simX[i,0])
            VX.append(simX[i,1])
            Y.append(simX[i,2])
            U.append(simU[i,0])
            Z.append(simX[i,4])
            VZ.append(simX[i,5])
            x_o.append(self.posx[i])
            y_o.append(self.posy[i])
            z_o.append(self.posz[i])
            vx_o.append(self.velx[i])
            vy_o.append(self.vely[i])
            vz_o.append(self.velz[i])
            ax_o.append(self.afx[i])
            ay_o.append(self.afy[i])
            az_o.append(self.afx[i])

            # update state
            # x_current = acados_integrator.get("x")
            # simX[i+1,:] = x_current
            
            k = k+1 # update the counter

    def update(self,frame):
        plt.cla()  # Clear the previous plot
        plt.plot(X[:frame], Y[:frame], label='Path')
        plt.plot(x_o[:frame], y_o[:frame], 'g-', label='Obstacle')
        plt.plot(X[frame-1], Y[frame-1], 'ro', label='Current Position')
        plt.plot(x_o[frame-1], y_o[frame-1], 'go', label='Current Obstacle Position')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()





if __name__ == '__main__':
    try:
		# Define the performance parameters here which starts the script
        N_horizon = 25
        T_horizon = 1.0#1.0/50
        Time = 60.00
        q=clover(N_horizon = N_horizon, T_horizon = T_horizon, Time = Time)
		
        q.main()
		#print(xa)
		#print(xf)
        
          
        # Plot logged data for analyses and debugging
        Nsim = int(Time * N_horizon / T_horizon)
        t = np.linspace(0.0, Nsim * T_horizon / N_horizon, Nsim)
        
        # Plot obstacle
        # Define center coordinates
        center = (8, 6)
        radius = 2.8

        # Create an array of angles from 0 to 2*pi
        theta = np.linspace(0, 2*np.pi, 100)

        # Calculate x and y coordinates of the circle
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)

        # Create a figure and axis
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, q.update, frames=len(X), interval=1.0)  # Change interval as needed (milliseconds)

        plt.show()

        # plt.figure(1)
        # plt.subplot(311)
        # plt.plot(t, X,'r',label='x-pos')
        # #plt.legend()
        # plt.grid(True)
        # plt.ylabel('pos[m]')
        # plt.xlabel('Time [s]')
        # plt.legend()
		# #plt.subplot(312)
		# #plt.plot(yf,'r',label='y-fol')
		# #plt.plot(ya,'b--',label='y-obs')
		# #plt.legend()
		# #plt.grid(True)
		# #plt.ylabel('Position [m]')
        # plt.subplot(312)
        # plt.plot(t, Y,'r',label='y-pos')
        # plt.ylabel('pos[m]')
        # plt.xlabel('Time [s]')
        # plt.legend()
        # plt.grid(True)
        # plt.subplot(313)
        # plt.plot(t, VX,'r',label='vx')
        # plt.ylabel('vel [m/s]')
        # plt.xlabel('Time [s]')
        # plt.legend()
        # plt.grid(True)

        # plt.figure(2)
        # plt.subplot(311)
        # plt.plot(t, U,'r')
        # plt.legend()
        # plt.grid(True)
        # plt.ylabel('yaw [deg]')
        # plt.xlabel('Time [s]')
        # plt.subplot(312)
        # plt.plot(t, Z,'r')
        # plt.grid(True)
        # plt.subplot(313)
        # plt.plot(t, VZ,'r')
        # plt.grid(True)
        
        # plt.figure(3)
        # plt.plot(X,Y)
        # plt.plot(x,y,'b-', label='Circle')
        # plt.plot(center[0], center[1], 'ro', label='Center (8,6)')
        # plt.plot(x_o,y_o,'g-',label='Obstacle')
        # plt.axis('equal')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.grid(True)
        # plt.show()
	
		
    except rospy.ROSInterruptException:
        pass
