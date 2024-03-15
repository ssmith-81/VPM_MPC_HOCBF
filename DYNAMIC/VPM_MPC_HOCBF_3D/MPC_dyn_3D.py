#!/usr/bin/python3

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from CLOVER_MODEL import export_clover_model
from FUNCTIONS import CloverCost, acados_settings
from casadi import SX, vertcat, sin, cos, norm_2, diag, sqrt 

import rospy
from clover import srv
from std_srvs.srv import Trigger
import math
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
from gazebo_msgs.msg import ModelState, LinkStates
import tf
from sensor_msgs.msg import Imu, LaserScan
from clover_callback import long_callback


from panel_functions import CLOVER_COMPONENTS, CLOVER_STREAM_GEOMETRIC_INTEGRAL, CLOVER_KUTTA, CLOVER_STREAMLINE, CLOVER_noOBSTACLE
from scipy.interpolate import griddata


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import path


import os

from clover import srv
from custom_message import srv as surf
from time import sleep
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *

import h5py  # use for data logging of larges sets of data
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from ellipse import LsqEllipse

# Custom message for object state publishing
from custom_message.msg import ObjectPub, ObjectExtend


from tf.transformations import euler_from_quaternion

rospy.init_node('MPC_3D')

# Set the use_sim_time parameter to true
rospy.set_param('use_sim_time', True)

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)

# Add in external position of target object service
object_loc = rospy.ServiceProxy('ObjectPose',surf.ObjectPose)

# Release service to allow for complex trajectory publishing i.e stopping navigate service publishing because you dont want two sources of publishing at the same time.
release = rospy.ServiceProxy('simple_offboard/release', Trigger)

PI_2 = math.pi/2

# Debugging and logging
xf = []  # This gathers the clover position
yf = []
zf = []
# position command from MPC dyamic output
xcom = []
ycom = []
zcom = []
YawF = []
YawC = []

# readings obstacle (for the cylinder)
xac = []
yac = []

# obstacle readings (for the second cylinder)
xap = []
yap = []

#ellipse estimation from lidar data
a_fit = []
b_fit = []
theta = []
xc = []# for the ellipse
yc = []
xcp = [] # for the prism
ycp = []
a_fitp = []
b_fitp = []
thetap = []
# drone dynamics Correspond to obstacle 1
x_ce = [] # these ones are for logging position/velocity corresponding to the lidar measurements
y_ce = [] # so we can calculate psi_0 and psi_1 offline or after the simulation
z_ce = []
vx_ce = []
vy_ce = []
vz_ce = []
# obstacle 1
xc_obs = []
vx_obsc = []
yc_obs = []
vy_obsc = []
zc_obs = []
vz_obsc = []
# Correspond to obstacle 2
x_pe = [] # these ones are for logging position/velocity corresponding to the lidar measurements
y_pe = [] # so we can calculate psi_0 and psi_1 offline or after the simulation
z_pe = []
vx_pe = []
vy_pe = []
vz_pe = []
# obstacle 2
xp_obs = []
vx_obsp = []
yp_obs = []
vy_obsp = []
zp_obs = []
vz_obsp = []

# readings of the non-modified/original obstacle1
xa_orig1 = []
ya_orig1 = []
# modified lidar obstacle 1
xa1 = []
ya1 = []
x_cur1 = []
y_cur1 = []
yaw1 = []
time1 = []

# readings of the non-modified/original obstacle1
xa_orig2 = []
ya_orig2 = []
# modified lidar obstacle 1
xa2 = []
ya2 = []
x_cur2 = []
y_cur2 = []
yaw2 = []
time2  =[]


# Analyze control input (see if error is being minimized )
velfx=[]
velfy=[]
velfz = []
# velocity command from MPC dynamic output
velcx=[]
velcy=[]
velcz=[]
# log velocity command from the VPM velocity field
uVPM = []
vVPM = []

# updated velocity field plot
nGridX = 20  # 20                                                         # X-grid for streamlines and contours
nGridY = 20  # 20                                                       # Y-grid for streamlines and contours
x_field = np.zeros((nGridX, nGridY))# np.zeros((30, 30))
y_field = np.zeros((nGridX, nGridY)) # np.zeros((30, 30))
u_field = np.zeros((nGridX, nGridY))
v_field = np.zeros((nGridX, nGridY))
lidar_x = []
lidar_y = []

# log the invarient sets for cylinder
psi_0 = []
psi_1 = []
r_cyl = []
timec = [] # log the time for plotting in matlab also 
# for prism
psi_0p = []
psi_1p = []
r_rec = []
timep = [] # log the time for plotting in matlab also 

U_infx = []
V_infy=[]
evx=[]
evy=[]
eyaw=[]

# Log the time for the position control variables
time_now = []

# Global lof variables
X = []
VX = []
Y = []
Ux = []
Uy = []
Uz = []
Z = []
VZ = []

# Get the path to the directory containing the Python script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Create a HDF5 file name
# Open the HDF5 file globally
file_name = 'mpc_dynamic_test7.h5' # 'mpc_static.h5'

# Construct the absolute path to the HDF5 file
absolute_file_path = os.path.join(script_dir, file_name)



 # Open the HDF5 file for writing
with h5py.File(absolute_file_path, 'a') as hf:

	class vortex:

		def VPM_calc(self,xa,ya,x_clover,y_clover, yaw, U_inf, V_inf, g_source, g_sink, xs, ys, xsi, ysi, g_clover,lidar_angles):

			# x_local = np.array(xa)
			# y_local = np.array(ya)

			# x_local_orig = np.array(xa)
			# y_local_orig = np.array(ya)

			x_local = xa[0]
			y_local = ya[0]

			x_local_orig = xa[0]
			y_local_orig = ya[0]

			yaw = yaw[0]

			x_clover = x_clover[0]
			y_clover = y_clover[0]

			# Kutta condition flag (decide which one to use)
			self.flagKutta = np.array([0, 0, 0,1])
            # position 0 is for smooth flow off edge for closed object
            # position 1 is for smooth flow off ege for closed object using point extended into space a small amount
            # position 2 is for smooth flow off the edge of a non closed i.e. sail detection with lidar
            # position 3 is for smooth flow off an extended edge of a non closed i.e. sail detection with lidar. Adds safety factor to direct flow away from object

			#---------------- Safety factor-----------------------------------------------
			# put a safety factor on the detected obstacle
			# Reduce the range by a scaling factor beta for each real range (set as diameter of the clover)
			beta = 2.5 # Scale object and shift
			# Combine xdata and ydata into a single array of points
			points = np.column_stack((x_local, y_local))

			# Find the point closest to the origin
			min_distance_index = np.argmin(np.linalg.norm(points, axis=1))
			closest_point = points[min_distance_index]

			# Step 2: Shift all points so that the closest point becomes the origin
			shifted_points = points - closest_point

			# Step 3: Scale all points by a factor beta
			scaled_points = shifted_points * beta

			# Step 4: Shift all points back to their original positions
			final_points = scaled_points + closest_point

			# Calculate the distance to move the closest point
			desired_distance = 0.2 # was 0.5

			# Calculate the current distance to the origin for the closest point
			current_distance = np.linalg.norm(closest_point)

			# Calculate the unit vector in the direction of the closest point
			unit_vector = closest_point / current_distance

			# Calculate the new position for the closest point
			new_closest_point = unit_vector * (current_distance - desired_distance)

			# Calculate the difference vector
			shift_vector = closest_point - new_closest_point

			# Shift all points including the closest point
			shifted_points = final_points - shift_vector


			# translate the shape equally to the origin (To clover)
			x_local = shifted_points[:,0]
			y_local = shifted_points[:,1]

			#------------------2D transformations--------------------
			# Homogenous transformation matrix for 2D
			R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) # rotation matrix
			T = np.vstack([np.hstack([R, np.array([[x_clover], [y_clover]])]),[0,0,1]]) # Homogenous transformation matrix

			# Lidar readings in homogenous coordinates
			readings_local = np.vstack([x_local, y_local, np.ones_like(x_local)])
			readings_local_orig = np.vstack([x_local_orig, y_local_orig, np.ones_like(x_local_orig)])

			# Transform all lidar readings to global coordinates
			readings_global = np.dot(T, readings_local)
			readings_global_orig = np.dot(T, readings_local_orig)

			# Extract the tranformed positions
			self.readings_global = readings_global[:2,:].T
			self.readings_global_orig = readings_global_orig[:2,:].T

			# Update the lidar detection readings for logging below
			self.xa = self.readings_global[:,0].T
			self.ya = self.readings_global[:,1].T

			self.xa_orig = self.readings_global_orig[:,0].T
			self.ya_orig = self.readings_global_orig[:,1].T

			# Filter out the inf values in the data point arrays
			self.xa = self.xa[np.isfinite(self.xa)]
			self.ya = self.ya[np.isfinite(self.ya)]
			self.xa_orig = self.xa_orig[np.isfinite(self.xa_orig)]
			self.ya_orig = self.ya_orig[np.isfinite(self.ya_orig)]


                #---------------------------------------------------------------------------------

			# Upate the number of panels
			self.n = len(self.xa)-1

			# in this case an obstacle was detected so apply panel method navigation

			# Get current state of this follower 
			# telem = get_telemetry(frame_id='map')

			# #-------------------- Offline Panel Calculations---------------------------------------------------

			#This function calculates the location of the control points as well as the
			#right hand side of the stream function equation:

			[xmid, ymid, dx, dy, Sj, phiD, rhs] = CLOVER_COMPONENTS(self.xa, self.ya, U_inf, V_inf, g_source, g_sink, xs, ys, xsi, ysi, self.n, g_clover, x_clover, y_clover)



			# Convert angles from [deg] to [rad]
			phi = np.deg2rad(phiD)  # Convert from [deg] to [rad]



			# Evaluate gemoetric integral matrix without the kutta condition equation
			I = CLOVER_STREAM_GEOMETRIC_INTEGRAL(xmid, ymid, self.xa, self.ya, phi, Sj, self.n)

			#-----------------------extended kutta condition---------------------------------------------------------
			# Extended point off of the end of the object for kutta condition
			# Calculate the extended edge point for the sail extended kutta condition
			ext_dist = 1.0


			finite_indices = np.where(np.isfinite(x_local))[0] # find where the indices are finite in the clover/local reference frame (this is being updated in the lidar function)
			ang = lidar_angles[finite_indices] # select the angles that are finite readings

			# Step 3: Compute the sums for left and right sides using array operations
			left_sum = np.sum(ang[ang > 0])
			right_sum = np.sum(ang[ang < 0])

			# Determine which side has the obstacle closer
			if left_sum > abs(right_sum):
				# print("Obstacle is more to the left.")
				#---------Object is more to the left of the clover when detecting-------------
				# this is intuitively backwards, you would think you would have to extend the kutta off the right side of the object
				# if the object was more to the left of the clover so we could go around the irght side of the object.

				directionVect = [self.xa[-1] - self.xa[-2], self.ya[-1] - self.ya[-2]] # off the end of the clockwise ending panel
				directionVect = directionVect / np.linalg.norm(directionVect)
				# normalize the direction vector

				# Calculate the coordinates of the extended point in the general frame
				extendedX = self.xa[-1] + ext_dist*directionVect[0]
				extendedY = self.ya[-1] + ext_dist*directionVect[1]

				#---------Log intuitive trail point for the plots in paper-------------
				q  = [self.xa[0] - self.xa[1], self.ya[0] - self.ya[1]] 
				q  = q / np.linalg.norm(q)

				int_Y = self.ya[0] + ext_dist*q[1]
				int_X = self.xa[0] + ext_dist*q[0]

				trail_intuitive = [int_X, int_Y]

			elif left_sum < abs(right_sum):
				# print("Obstacle is more to the right.")
				#-----------Object is more to the right of the clover when detecting------------
				# This is intuitively backwards, as you would think this would be the case for when the object is detected more in the left half 
				# of the clover...(not sure why this works this way). This kutta condition will have the flowlines go off the left side of the object
				directionVect = [self.xa[0] - self.xa[1], self.ya[0] - self.ya[1]] # off the end of the CCW/right ending panel
				directionVect = directionVect / np.linalg.norm(directionVect)
				extendedY = self.ya[0] + ext_dist*directionVect[1]
				extendedX = self.xa[0] + ext_dist*directionVect[0]

				#---------Log intuitive trail point for the plots in paper-------------
				q  = [self.xa[-1] - self.xa[-2], self.ya[-1] - self.ya[-2]]
				q  = q / np.linalg.norm(q)

				int_Y = self.ya[-1] + ext_dist*q[1]
				int_X = self.xa[-1] + ext_dist*q[0]

				trail_intuitive = [int_X, int_Y]
			else:
				# print("Obstacle is centered.")

				directionVect = [self.xa[0] - self.xa[1], self.ya[0] - self.ya[1]] # off the end of the CCW/right ending panel
				directionVect = directionVect / np.linalg.norm(directionVect)
				extendedY = self.ya[0] + ext_dist*directionVect[1]
				extendedX = self.xa[0] + ext_dist*directionVect[0]

				#---------Log other one-------------
				q  = [self.xa[-1] - self.xa[-2], self.ya[-1] - self.ya[-2]]
				q  = q / np.linalg.norm(q)

				int_Y = self.ya[-1] + ext_dist*q[1]
				int_X = self.xa[-1] + ext_dist*q[0]

				trail_intuitive = [int_X, int_Y]



			# coordinates of the translated point off the trailing edge
			trail_point = [extendedX, extendedY]
			#---------------------------------------------------------------------------------------------------------------------------------------------
			# Form the last line of the system of equations with the kutta condition 
			[I, rhs] = CLOVER_KUTTA(I, trail_point, self.xa, self.ya, phi, Sj, self.n, self.flagKutta, rhs, U_inf, V_inf, xs, ys, xsi, ysi, g_source, g_sink, g_clover, x_clover, y_clover)

			# calculating the vortex density (and stream function from kutta condition)
			# by solving linear equations given by
			g = np.linalg.solve(I, rhs.T)  # = gamma = Vxy/V_infinity for when we are using U_inf as the free flow magnitude
			# broken down into components (using cos(alpha), and sin(alpha)), and free flow is the only other RHS
			# contribution (no source or sink). It is equal to Vxy 
			# when we define the x and y (U_inf and V_inf) components seperately.


			# Path to figure out if grid point is inside polygon or not
			AF = np.vstack((self.xa,self.ya)).T
			AF_orig = np.vstack((self.xa_orig,self.ya_orig)).T
			#print(AF)
			afPath = path.Path(AF)
			afPath_orig = path.Path(AF_orig)

			# Grid parameters
			self.nGridX = 20;  # 20 is good                                                         # X-grid for streamlines and contours
			self.nGridY = 20;  # 20 is good                                                    # Y-grid for streamlines and contours
			self.xVals  = [-1, 21];  # ensured it is extended past the domain incase the clover leaves domain             # X-grid extents [min, max]
			self.yVals  = [-1, 21];  #-0.3;0.3                                                 # Y-grid extents [min, max]

			# Define Lidar range (we will set parameters to update grid resolution within detection range):
			self.lidar_range = 3.5 # [m]

			
			# Create an array of starting points for the streamlines
			x_range = np.linspace(0, 10, int((10-0)/0.5) + 1)
			y_range = np.linspace(0, 10, int((10-0)/0.5) + 1)

			x_1 = np.zeros(len(y_range))
			y_1 = np.zeros(len(x_range))
			Xsl = np.concatenate((x_1, x_range))
			Ysl = np.concatenate((np.flip(y_range), y_1))
			XYsl = np.vstack((Xsl,Ysl)).T

			# Generate the grid points
			Xgrid = np.linspace(self.xVals[0], self.xVals[1], self.nGridX)
			Ygrid = np.linspace(self.yVals[0], self.yVals[1], self.nGridY)
			self.XX, self.YY = np.meshgrid(Xgrid, Ygrid)

			self.Vxe = np.zeros((self.nGridX, self.nGridY))
			self.Vye = np.zeros((self.nGridX, self.nGridY))

			for m in range(nGridX):
				for n in range(nGridY):
					XP, YP = self.XX[m, n], self.YY[m, n]
					# XP, YP = self.X_mesh[m, n], self.Y_mesh[m, n]
					# Check if the current grid point corresponds to (xa, ya) or (xa_orig, ya_orig)
					if  afPath_orig.contains_points([[XP,YP]]):#afPath.contains_points([[XP,YP]]) or afPath_orig.contains_points([[XP,YP]]):
						self.Vxe[m, n] = 0
						self.Vye[m, n] = 0

					else:
						u, v = CLOVER_STREAMLINE(XP, YP, self.xa, self.ya, phi, g, Sj, U_inf, V_inf, xs, ys, xsi, ysi, g_source, g_sink, g_clover, x_clover, y_clover)
						# print(u)


						self.Vxe[m, n] = u

						self.Vye[m, n] = v


			# Flatten the grid point matices and velocity matrices into vectory arrays for the griddata function
			# Update the velocity field:
			self.XX_f = self.XX.flatten()
			self.YY_f = self.YY.flatten()
			self.Vxe_f = self.Vxe.flatten()
			self.Vye_f = self.Vye.flatten()


			# Return lists of ellipse parameters for each row of data
			return self.xa,self.ya, self.xa_orig, self.ya_orig, self.Vxe,self.Vye, extendedX, extendedY, int_X, int_Y

	lidar_prism = hf.create_group('Lidar_reading_prism')
	lidar_cylinder = hf.create_group('Lidar_reading_cylinder')

	class ellipse:

		def ellipse_calc(self, x_ce, vx_ce, y_ce, vy_ce,z_ce, vz_ce, xc_obs,vx_obsc,yc_obs,vy_obsc,zc_obs,vz_obsc, x_pe, vx_pe, y_pe, vy_pe,z_pe, vz_pe,xp_obs, vx_obsp,yp_obs,vy_obsp,zp_obs,vz_obsp):

			# This function is used to calculate the psi 0 ans 1 values for this 3D simulation case

			#--------------Obstacle parameters-----------------------------
			self.SF = 1.0 # safety factor distance from the obstcle (set as the width of the Clover)
			self.SFp = 1.2 # safety factor for prism
			self.cyl_rad = 1.5 # [m] radius of the cylinder
			self.rec_rad = 1.25 # [m] half of the longest side of the prism

			#--------------------------------------------------------------

			# Doing the cylinder and prism seperately because the prism is giving straneg results on psi_1

			
			# Iterate over rows of data (for the second obstacle)
			for i in range(len(xc_obs)): # x_pe
				

				#----------- Calculate delta_p, delta_v, and delta_a-----------------------------------
				delta_p = np.array([x_pe[i] - xp_obs[i], y_pe[i]-yp_obs[i], z_pe[i] - zp_obs[i]])
				delta_v = np.array([vx_pe[i]-vx_obsp[i], vy_pe[i]-vy_obsp[i],vz_pe[i]-vz_obsp[i]]) # static obstacle assumption for now

				# Calculate norms
				norm_delta_p = np.linalg.norm(delta_p, ord=2)  # Euclidean norm
				norm_delta_v = np.linalg.norm(delta_v, ord=2)  # Euclidean norm

				# constants
				q1 = 15#15
				q2 = 10#10
				#print(max(max(ellipse_model.params[2], ellipse_model.params[3]),0.8)) # woow, some pretty unstacle shapes come out of this on the prism
				r = self.SF + 1.5#max(ellipse_model.params[2], ellipse_model.params[3])#self.cyl_rad #max(max(ellipse_model.params[2], ellipse_model.params[3]),0.8) # make sure it is at least giving 0.8, where the max prism radius is 1.25 i believe

				self.psi_0p = norm_delta_p - r
				self.psi_1p = (np.dot(delta_p, delta_v)) / norm_delta_p + q1*(self.psi_0p)

				psi_0p.append(self.psi_0p)
				psi_1p.append(self.psi_1p)
				r_cyl.append(r)
				# Extract parameters of the fitted ellipse
				# xcp.append(ellipse_model.params[0])
				# ycp.append(ellipse_model.params[1])
				# a_fitp.append(ellipse_model.params[2])
				# b_fitp.append(ellipse_model.params[3])
				# thetap.append(ellipse_model.params[4])

		#--------------------------------------------------------------------------------------------

			# Iterate over rows of data (for the first obstacle)
			for i in range(len(xp_obs)): # x_ce
				
				#----------- Calculate delta_p, delta_v, and delta_a-----------------------------------
				delta_p = np.array([x_ce[i]-xc_obs[i], y_ce[i]-yc_obs[i],z_ce[i] - zc_obs[i]])
				delta_v = np.array([vx_ce[i]-vx_obsc[i], vy_ce[i]-vy_obsc[i], vz_ce[i]-vz_obsc[i]]) # static obstacle assumption for now

				# Calculate norms
				norm_delta_p = np.linalg.norm(delta_p, ord=2)  # Euclidean norm
				norm_delta_v = np.linalg.norm(delta_v, ord=2)  # Euclidean norm

				# constants
				q1 = 15#15
				q2 = 10#10
				r = self.SF + 1.5#max(ellipse_model.params[2], ellipse_model.params[3])

				self.psi_0 = norm_delta_p - r
				self.psi_1 = (np.dot(delta_p, delta_v)) / norm_delta_p + q1*(self.psi_0)

				psi_0.append(self.psi_0)
				psi_1.append(self.psi_1)
				r_rec.append(r)
		#--------------------------------------------------------------------------------------------


				# Extract parameters of the fitted ellipse
				# xc.append(ellipse_model.params[0])
				# yc.append(ellipse_model.params[1])
				# a_fit.append(ellipse_model.params[2])
				# b_fit.append(ellipse_model.params[3])
				# theta.append(ellipse_model.params[4])
			
			# Return lists of ellipse parameters for each row of data
			return psi_0, psi_1, r_rec, psi_0p, psi_1p, r_cyl


	class clover:

		def __init__(self, FLIGHT_ALTITUDE = 2.2, RATE = 50, RADIUS = 3.5, V_des = 0.6, N_horizon=25, T_horizon=5.0, REF_FRAME = 'map'): # rate = 50hz radius = 5m cycle_s = 25 N_h = 25, T_h = 5

			# Note on N_horizon and T_horizon: with how I have things now, set T_horizon higher to react quicker to obstacles, but if you increase, this increases the time ahead of the first shooting node some 
			# which is which may increase speed some. N_horizon, increase to slow drone down some and react faster to obstacles. Lower, increases time to first shoting node and increases speed

			# Define the sink location and strength
			self.g_sink = 2.4
			self.xsi = 20 # 20
			self.ysi = 20 # 20

			# Define the source strength and location
			self.g_source = 0.9
			self.xs = -0.1
			self.ys = -0.1
			# self.xs = 5
			# self.ys = 2.5


			# Define the source strength placed on the actual drone
			self.g_clover = 0.0

			# Could possible just put a source on the drone and not bother with the two different sources


			# Free flow constant
			self.U_inf = 0
			self.V_inf = 0

			self.alpha = 70*(math.pi/180)

		# iNITIALIZE INPUT VELOCITY VALUES
			self.u = 0
			self.v = 0

			# Define the max velocity allowed for the Clover
			self.vel_max = 0.85 # [m/s] 0.65

			# #-------------------- Offline Panel Calculations---------------------------------------------------

			# An object was not detected yet, so use the source and sink for velocity based navigation

			## Compute Streamlines with stream function velocity equation
			# Too many gridpoints is not good, it will cause the control loop to run too slow
			# Grid parameters
			self.nGridX = 20;  # 20 is good                                                         # X-grid for streamlines and contours
			self.nGridY = 20;  # 20 is good                                                    # Y-grid for streamlines and contours
			self.xVals  = [-1, 21];  # ensured it is extended past the domain incase the clover leaves domain             # X-grid extents [min, max]
			self.yVals  = [-1, 21];  #-0.3;0.3                                                 # Y-grid extents [min, max]

			# Define Lidar range (we will set parameters to update grid resolution within detection range):
			self.lidar_range = 3.5 # [m]

			
			# Create an array of starting points for the streamlines
			x_range = np.linspace(0, 10, int((10-0)/0.5) + 1)
			y_range = np.linspace(0, 10, int((10-0)/0.5) + 1)

			x_1 = np.zeros(len(y_range))
			y_1 = np.zeros(len(x_range))
			Xsl = np.concatenate((x_1, x_range))
			Ysl = np.concatenate((np.flip(y_range), y_1))
			XYsl = np.vstack((Xsl,Ysl)).T

			# Generate the grid points
			Xgrid = np.linspace(self.xVals[0], self.xVals[1], self.nGridX)
			Ygrid = np.linspace(self.yVals[0], self.yVals[1], self.nGridY)
			self.XX, self.YY = np.meshgrid(Xgrid, Ygrid)

			self.Vxe = np.zeros((self.nGridX, self.nGridY))
			self.Vye = np.zeros((self.nGridX, self.nGridY))

			# Get the current state/position of th clover for its source
			telem = get_telemetry(frame_id='map')


			for m in range(self.nGridX):
				for n in range(self.nGridY):
					XP, YP = self.XX[m, n], self.YY[m, n]
					u, v = CLOVER_noOBSTACLE(XP, YP, self.U_inf, self.V_inf, self.xs, self.ys, self.xsi, self.ysi, self.g_source, self.g_sink, self.g_clover, telem.x, telem.y)

					self.Vxe[m, n] = u
					self.Vye[m, n] = v

			# Flatten the grid point matices and velocity matrices into vector arrays for the griddata function
			self.XX_f = self.XX.flatten()
			self.YY_f = self.YY.flatten()
			self.Vxe_f = self.Vxe.flatten()
			self.Vye_f = self.Vye.flatten()

			#-------------External LOG------------------
            # Create a group to store velocity field for thisiteration
			iteration_group = hf.create_group('Initial_Field')
			iteration_group.create_dataset('XX', data=self.XX)
			iteration_group.create_dataset('YY', data=self.YY)
			iteration_group.create_dataset('Vxe', data=self.Vxe)
			iteration_group.create_dataset('Vye', data=self.Vye)
            #------------------------------------------

			
			
			# Publisher which will publish to the topic '/mavros/setpoint_velocity/cmd_vel'.
			self.velocity_publisher = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',TwistStamped, queue_size=10)

			#-------MPC formulation----------------------------------------
			# MPC variables
			self.N_horizon = N_horizon # Define prediction horizone in terms of optimization intervals
			self.T_horizon = T_horizon # Define the prediction horizon in terms of time (s) --> Limits time and improves efficiency
			# This will have time step within prediction horizon as dt = T/N, would probably like it to be close to
			# self.CYCLE_S/self.STEPS = dt
			self.dt              =   self.T_horizon/self.N_horizon

			# Compute the prediction horizon length in terms of steps in the reference trajectory
			# self.N_steps = int(self.T_horizon*self.RATE)
			# self.dt_ref = int(self.N_steps/self.N_horizon) # This is the amount of steps ahead within the reference trajectory array, per iteration in the prediction horizon

			# load model
			self.acados_solver, acados_integrator, model = acados_settings(self.N_horizon, self.T_horizon)
			# dimensions
			self.nx = model.x.size()[0]
			self.nu = model.u.size()[0]
			ny = self.nx + self.nu

			#---------------------------------------------------------------------------------------------------
			
			
			# global static variables
			self.FLIGHT_ALTITUDE = FLIGHT_ALTITUDE          # fgdf
			# self.RATE            = RATE                     # loop rate hz
			self.RATE            =  self.N_horizon/self.T_horizon # i think this makes sense, because dt = self.T_horizon/self.N_horizon
			self.FRAME           = REF_FRAME                # Reference frame for complex trajectory tracking
			self.CYCLE_S         = 15 #distance / self.V_des     # time to complete linear trajectory in seconds
			self.STEPS           = int( self.CYCLE_S * self.RATE ) # Total number of steps in the trajectory

			self.last_timestamp = None # make sure there is a timestamp
			

			i    = 0
			dt   = 1.0/self.RATE # = dt = self.T_horizon/self.N_horizon

			

			#--------------Obstacle parameters-----------------------------
			self.SF = 1.0 # safety factor distance from the obstcle (set as the width of the Clover)
			self.cyl_rad = 1.5 # [m] radius of the cylinder
			# Center of the cylinder location for experiment
			self.x_cyl = 6.0
			self.y_cyl = 5.1
			self.SFp = 0.9 # safety factor for prism
			self.rec_rad = 1.25 # [m] half of the longest side of the prism
			# Center of the prism for experiment
			self.x_rec = 11.5
			self.y_rec = 15.2

			#--------------------------------------------------------------
			
			
			# Publisher which will publish to the topic '/mavros/setpoint_raw/local'.
			self.publisher = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
			
			# Subscribe to drone state
			self.state = rospy.Subscriber('mavros/state', State, self.updateState)

			# Subscribe directly to the ground truth
			self.ground_truth = rospy.Subscriber('gazebo/link_states', LinkStates, self.link_states_callback)

			# Use the data coming in through this subscription, subscribing is faster than service, and we are using ground_truth from gazebo for better pose estimations:
			self.pose_call = rospy.Subscriber('/mavros/local_position/velocity_local',TwistStamped, self.controller)

			# Generate the array of lidar angles
			self.lidar_angles = np.linspace(-180*(math.pi/180), 180*(math.pi/180), 360) # Adjust this number based on number defined in XACRO!!!!!

			# Initialize the ranges array for the controller as no object detected (i.e. an array of inf)
			self.obs_detect = np.full_like(self.lidar_angles, np.inf)

			# Subscribe to the Lidar readings
			self.lidar = rospy.Subscriber('/ray_scan',LaserScan,self.lidar_read)

			 # Subscribe to the observer estimation publishing:
			self.AHOSMO1 = rospy.Subscriber('/AHOSMO_est1', ObjectExtend, self.AHOSMO_callback1)
			self.AHOSMO2 = rospy.Subscriber('/AHOSMO_est2', ObjectExtend, self.AHOSMO_callback2)

			# Set a flag, that will be used as a logic operator to adjust how the HOCBF is set, which depends on whether an obstacle is detected or not
			self.object_detect = False

			# Set this flag to calculate the ellipse shape on the first detection (cant iteraivly do it in this envirnment as the least squared
			# calculation at each iteration is too computationally intensive. It sucks up all the CPU usage and slows the lidar callback loop down)
			self.ellipse_first = False

			# This flag will be used to determine which obstacle is coming next and what pose to send to the MPC solver ACADOS
			self.obstacle_counter = 0 

			#-----Define velocity field plot logging variables--------------
			self.count = True

			self.current_state = State()
			self.rate = rospy.Rate(20) # 20



		def updateState(self, msg):
			self.current_state = msg

		def link_states_callback(self, msg):
			try:
				index = msg.name.index('clover::base_link')
			except ValueError:

				return

			self.clover_pose = msg.pose[index]

		def AHOSMO_callback1(self, data):
		# Probably need to set a condition when there are no estimations etc ( dont have to
		# this is already handles in the AHOSMO module, where estimations are only provided when something is detected
		# and if something isnt detected, the trivial case is publish in this script)
			self.obs_x1 = data.x
			self.obs_y1 = data.y
			self.obs_z1 = data.z
			self.obs_vx1 = data.vx
			self.obs_vy1 = data.vy
			self.obs_vz1 = data.vz
			self.obs_ax1 = data.ax
			self.obs_ay1 = data.ay
			self.obs_az1 = data.az
			


		def AHOSMO_callback2(self, data):
            # Probably need to set a condition when there are no estimations etc ( dont have to
            # this is already handles in the AHOSMO module, where estimations are only provided when something is detected
            # and if something isnt detected, the trivial case is publish in this script)
			self.obs_x2 = data.x
			self.obs_y2 = data.y
			self.obs_z2 = data.z
			self.obs_vx2 = data.vx
			self.obs_vy2 = data.vy
			self.obs_vz2 = data.vz
			self.obs_ax2 = data.ax
			self.obs_ay2 = data.ay
			self.obs_az2 = data.az

		@long_callback
		def lidar_read(self,data):
			

			# Read the current timestamp of the readings
			self.lidar_timestamp = data.header.stamp.to_sec()

			# Update the obstacle detect array
			self.obs_detect = data.ranges

			# Get current state of this follower 
			# telem = get_telemetry(frame_id='map')
			x_clover = self.clover_pose.position.x
			y_clover = self.clover_pose.position.y
			z_clover = self.clover_pose.position.z
			quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
			euler_angles = euler_from_quaternion(quaternion)
			roll = euler_angles[2] #+ math.pi
			yaw = -euler_angles[0]+math.pi 
			pitch = euler_angles[1]

			

			# Ensure there are actually lidar readings, no point in doing calculations if
			# nothing is detected:
			# Need to add a lidar detection threshold for ellipsoid estimation, so if we have like 1-2 or low detection we could get bad results
			if x_clover < 13 and y_clover < 13:

				telem = get_telemetry(frame_id='map')
				current_time = rospy.Time.now()
				# Append the Clover position and veloicty in the first quadrant corresponding to obstacle 1
				x_ce.append(self.clover_pose.position.x) # current clover state of reading
				vx_ce.append(telem.vx)
				y_ce.append(self.clover_pose.position.y)
				vy_ce.append(telem.vy)
				z_ce.append(self.clover_pose.position.z)
				vz_ce.append(telem.vz)
				# append the obstacle dynamics
				xc_obs.append(self.obs_x1[0])
				vx_obsc.append(self.obs_vx1[0])
				yc_obs.append(self.obs_y1[0])
				vy_obsc.append(self.obs_vy1[0])
				zc_obs.append(self.obs_z1[0])
				vz_obsc.append(self.obs_vz1[0])
				
				timec.append(current_time.to_sec())

			else:

				# append clover dynamics for calculations for second obstacle in the second quadrant
				telem = get_telemetry(frame_id='map')
				current_time = rospy.Time.now()
				x_pe.append(self.clover_pose.position.x)
				vx_pe.append(telem.vx)
				y_pe.append(self.clover_pose.position.y)
				vy_pe.append(telem.vy)
				z_pe.append(self.clover_pose.position.z)
				vz_pe.append(telem.vz)
				# append the obstacle dynamics
				xp_obs.append(self.obs_x2[0])
				vx_obsp.append(self.obs_vx2[0])
				yp_obs.append(self.obs_y2[0])
				vy_obsp.append(self.obs_vy2[0])
				zp_obs.append(self.obs_z2[0])
				vz_obsp.append(self.obs_vz2[0])
				timep.append(current_time.to_sec())


		def controller(self,data):

			current_timestamp = data.header.stamp.to_sec()

			if self.last_timestamp is not None:


				# Get current state of this follower 
				# telem = get_telemetry(frame_id='map')
				x_clover = self.clover_pose.position.x
				y_clover = self.clover_pose.position.y
				z_clover = self.clover_pose.position.z
				quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
				euler_angles = euler_from_quaternion(quaternion)
				roll = euler_angles[2] #+ math.pi
				yaw = -euler_angles[0]+math.pi 
				pitch = euler_angles[1]



				# #-------------------- Offline Panel Calculations---------------------------------------------------



				# Complete contributions from pre-computed grid distribution
				u = griddata((self.XX_f, self.YY_f),self.Vxe_f,(x_clover,y_clover),method='nearest') #+ self.u_inf #+ u_source #+ u_inf 'linear' is way too computationally expensive
				v = griddata((self.XX_f, self.YY_f),self.Vye_f,(x_clover,y_clover),method='nearest') #+self.v_inf#+ v_source #+self.v_inf #+ v_source #+ v_inf
				# u=0.3
				# v=0.3


				# # Complete contributions from pre-computed grid distribution
				# self.u = u
				# self.v = v

				# normalize velocities
				vec = np.array([[u],[v]],dtype=float) 
				magnitude = math.sqrt(u**2 + v**2)

				if magnitude > 0:
					norm_vel = (vec/magnitude)*self.vel_max
				else:
					norm_vel = np.zeros_like(vec)

				self.u = norm_vel[0,0]
				self.v = norm_vel[1,0]
				
				# determine the yaw
				self.omega = math.atan2(self.v,self.u)


			# Update last_timestamp for the next callback
			self.last_timestamp = current_timestamp

				

		def main(self):
			

			# Wait for 3 seconds
			rospy.sleep(3)
			
			# Takeoff with Clovers navigate function
			navigate(x=0.5, y=0.5, z=self.FLIGHT_ALTITUDE, yaw=float('nan'), speed=0.5, frame_id = 'map', auto_arm = True)
			
			# Give the Clover time to reach the takeoff altitude
			rospy.sleep(15)

			# Define object that will be published
			target = PositionTarget()
			rr = rospy.Rate(self.RATE)
			k=0    # Define a counter for the reference arrays
			

			# update initial condition (current pose of MPC problem)
			telem = get_telemetry(frame_id = 'map')
			x_clover = self.clover_pose.position.x
			y_clover = self.clover_pose.position.y
			z_clover = self.clover_pose.position.z
			quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
			euler_angles = euler_from_quaternion(quaternion)
			roll = euler_angles[2] #+ math.pi
			yaw = -euler_angles[0]+math.pi 
			pitch = euler_angles[1]

			x0 = np.array([x_clover, telem.vx, y_clover, telem.vy, z_clover, telem.vz])
			self.acados_solver.set(0, "lbx", x0) # Update the zero shooting node lower bound position
			self.acados_solver.set(0, "ubx", x0) # update the zero shooting node upper bound position

			release() # stop navigate service from publishing

			while not rospy.is_shutdown():
			
				# Trajectory publishing-----------------------------------------------
				target.header.frame_id = self.FRAME  # Define the frame that will be used
				
				target.coordinate_frame = 1 #MAV_FRAME_LOCAL_NED  # =1
				
				# Dont use velocity and position at the same time, of the Clover deviates from position then, flight issues will happen
				# because of the mismatch in position and velocity.
				target.type_mask = 8+16+32+64+128+256+2048 # Use everything! 1024-> forget yaw
				#target.type_mask =  3576 # Use only position #POSITION_TARGET_TYPEMASK_VX_IGNORE | POSITION_TARGET_TYPEMASK_VY_IGNORE | POSITION_TARGET_TYPEMASK_VZ_IGNORE | POSITION_TARGET_TYPEMASK_AX_IGNORE | POSITION_TARGET_TYPEMASK_AY_IGNORE |POSITION_TARGET_TYPEMASK_AZ_IGNORE | POSITION_TARGET_TYPEMASK_FORCE_IGNORE | POSITION_TARGET_TYPEMASK_YAW_IGNORE | POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE # = 3576
				#target.type_mask =  3520 # Use position and velocity
				#target.type_mask =  3072 # Use position, velocity, and acceleration
				#target.type_mask =  2048 # Use position, velocity, acceleration, and yaw
				# uint16 IGNORE_VX = 8 # Velocity vector ignore flags
				# uint16 IGNORE_VY = 16
				# uint16 IGNORE_VZ = 32

					
				
				# update reference
				for j in range(self.N_horizon): # Up to N-1
									
					# yref = np.array([self.xsi, self.u, self.ysi, self.v, self.FLIGHT_ALTITUDE, 0, 0, 0, 0])
					yref = np.array([0, self.u, 0, self.v, self.FLIGHT_ALTITUDE, 0, 0, 0, 0])
					
					self.acados_solver.set(j, "yref", yref)

					# Check the drone's position and perform actions accordingly (within the first obstacle domain)
					if x_clover < 13 and y_clover < 13:

						r = self.SF + self.cyl_rad

						# We are passing all of the needed position, velocity, and acceleration values from the set state MPC
						self.acados_solver.set(j, "p", np.array([self.obs_x1[j],self.obs_vx1[j],self.obs_ax1[j],self.obs_y1[j],self.obs_vy1[j],self.obs_ay1[j],self.obs_z1[j],self.obs_vz1[j],self.obs_az1[j], r])) # set the obstacle dynamics [x,vx,ax,y,vy,ay,z,vz,az] dynamic obstacle
					
					# Check the drone's position and perform actions accordingly (within the second obstacle domain)
					elif x_clover >= 13 and y_clover >= 13:
					# 	# Set the distance from the obstacle
						
						r = self.SF + self.cyl_rad
						
						# self.acados_solver.set(j, "p", np.array([self.obs_x2[j],self.obs_vx2[j],self.obs_ax2[j],self.obs_y2[j],self.obs_vy2[j],self.obs_ay2[j],self.obs_z2[j],self.obs_vz2[j],self.obs_az2[j], r])) # set the obstacle dynamics [x,vx,ax,y,vy,ay,z,vz,az] dynamic obstacle
						self.acados_solver.set(j, "p", np.array([40,0,0,40,0,0,40,0,0, r])) # Assuming a static obstacle far away
					else:
   					 # Perform some default action if none of the conditions are met
					
						# 	# Set the distance from the obstacle
						r = 2.5
						
						self.acados_solver.set(j, "p", np.array([40,0,0,40,0,0,40,0,0, r])) # Assuming a static obstacle far away

					 # 	
				# Set the terminal reference state
				yref_N = np.array([0,self.u,0,self.v,self.FLIGHT_ALTITUDE,0]) # terminal components
				
				self.acados_solver.set(self.N_horizon, "yref", yref_N)
				
				# Solve ocp
				status = self.acados_solver.solve()

				# get solution
				x0 = self.acados_solver.get(0, "x")
				u0 = self.acados_solver.get(0, "u")

				# update initial condition
				x1 = self.acados_solver.get(1, "x")
				x8 = self.acados_solver.get(8, "x") # use this so it reacts faster to changes in environment
		

				# Gather position for publishing
				target.position.x = x1[0]
				target.position.y = x1[2]
				target.position.z = x1[4]
				
				
				# Gather velocity for publishing
				target.velocity.x = x0[1]
				target.velocity.y = x0[3]
				target.velocity.z = 0
				
				# Gather acceleration for publishing
				# target.acceleration_or_force.x = afx[k]
				# target.acceleration_or_force.y = afy[k]
				# target.acceleration_or_force.z = afz[k]

				# Feedforward acceleration setpoints as well
				target.acceleration_or_force.x = u0[0] #+ afx[k]
				target.acceleration_or_force.y = u0[1] #+ afy[k]
				target.acceleration_or_force.z = u0[2] #+ afz[k]
				
				# Might not want to consider yaw and yawrate as it will intefere with MPC and its constraints
				# Gather yaw for publishing
				# target.yaw = self.yawc[k]
				self.omega = math.atan2(x0[3],x0[1])
				target.yaw = self.omega
				
				# Gather yaw rate for publishing
				# target.yaw_rate = self.yaw_ratec[k]
				
				
				self.publisher.publish(target)

				
				# update initial condition (current pose of MPC problem)
				telem = get_telemetry(frame_id = 'map')
				x_clover = self.clover_pose.position.x
				y_clover = self.clover_pose.position.y
				z_clover = self.clover_pose.position.z
				quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
				euler_angles = euler_from_quaternion(quaternion)
				roll = euler_angles[2] #+ math.pi
				yaw = -euler_angles[0]+math.pi 
				# x_new = np.array([x_clover, x1[1], y_clover, x1[3], z_clover, x1[5]])
				x_new = np.array([x_clover, telem.vx, y_clover, telem.vy, z_clover, telem.vz])
				# self.acados_solver.set(0, "lbx", x_new) # Update the zero shooting node position
				# self.acados_solver.set(0, "ubx", x_new)
				self.acados_solver.set(0, "lbx", x1) # Update the zero shooting node position
				self.acados_solver.set(0, "ubx", x1)


				# logging/debugging
				xf.append(x_clover)
				yf.append(y_clover)
				zf.append(z_clover)
				xcom.append(x0[0])
				ycom.append(x0[2])
				zcom.append(x0[4])
				# evx.append(self.u-x_clover)
				# evy.append(self.v-telem.vy)
				eyaw.append(self.omega-yaw)
				YawC.append(self.omega*(180/math.pi))
				YawF.append(yaw*(180/math.pi))
				#xdispf.append(self.d_xi)
				#ydispf.append(self.d_yi)
				velfx.append(telem.vx)
				velfy.append(telem.vy)
				velfz.append(telem.vz)
				velcx.append(x0[1])
				velcy.append(x0[3])
				velcz.append(x0[5])
				uVPM.append(self.u)
				vVPM.append(self.v)
				# U_infx.append(self.U_inf)
				# V_infy.append(self.V_inf)
				Ux.append(u0[0])
				Uy.append(u0[1])
				Uz.append(u0[2])

				# Get the actual time:
				current_time = rospy.Time.now()
				time_now.append(current_time.to_sec())

			
				#set_position(x=posx[k], y=posy[k], z=posz[k],frame_id='aruco_map')
				#set_velocity(vx=velx[k], vy=vely[k], vz=velz[k],frame_id='aruco_map')#,yaw = yawc[i]) 
				#set_attitude(yaw = yawc[k],pitch = pitchc[k], roll = rollc[k], thrust = float('nan'),frame_id='aruco_map')
				#set_rates(yaw_rate = yaw_ratec[k],thrust = float('nan'))
			
			
				
				k = k+1
				# if k >= self.STEPS: 
				# 	navigate(x=0, y=0, z=self.FLIGHT_ALTITUDE, yaw=float('nan'), speed=0.5, frame_id = 'map')
				# 	rospy.sleep(5)
				# 	break
				if math.sqrt((x_clover-self.xsi) ** 2 + (y_clover-self.ysi) ** 2) < 0.6: # 0.4
					set_position(frame_id='body')
					# navigate(x=x_clover,y=y_clover,z=self.FLIGHT_ALTITUDE, yaw=float('nan'), speed=0.2, frame_id = self.FRAME)
					
					break
				
				rr.sleep()
				

			# Wait for 6 seconds
			rospy.sleep(6)
			# Perform landing
			
			land()
			
		
			

	if __name__ == '__main__':
		try:
			q=clover()
			
			q.main()#x=0,y=0,z=1,frame_id='aruco_map')

		# Define some preliminary plot limits for streamplot
			xVals  = [-1, 21];  # -0.5; 1.5                                                  # X-grid extents [min, max]
			yVals  = [-1, 21];  #-0.3;0.3                                                 # Y-grid extents [min, max]
			# Create an array of starting points for the streamlines
			# x_range = np.linspace(0, 20, int((20-0)/0.75) + 1)
			# y_range = np.linspace(0, 20, int((20-0)/0.75) + 1)
			# x_1 = np.zeros(len(y_range))
			# y_1 = np.zeros(len(x_range))

			x_range = np.linspace(2, 8, int((8-2)/0.5) + 1)
			y_range = np.linspace(2, 8, int((8-2)/0.5) + 1)
			x_1 = np.ones(len(y_range))*2
			y_1 = np.ones(len(x_range))*2
			# Convert list fields to arrays
			# x_field = np.array(x_field).reshape(2,2)
			# y_field = np.array(y_field).reshape(2,2)
			# u_field = np.array(u_field).reshape(2,2)
			# v_field = np.array(v_field).reshape(2,2)

			
			Xsl = np.concatenate((x_1, x_range))
			Ysl = np.concatenate((np.flip(y_range), y_1))
			XYsl = np.vstack((Xsl,Ysl)).T

			# Send and retrieve the ellipse, and psi_0 and psi_1 values

			w = ellipse()

			psi_0, psi_1, r_rec, psi_0p, psi_1p, r_cyl = w.ellipse_calc(x_ce, vx_ce, y_ce, vy_ce,z_ce, vz_ce, xc_obs,vx_obsc,yc_obs,vy_obsc,zc_obs,vz_obsc, x_pe, vx_pe, y_pe, vy_pe,z_pe, vz_pe,xp_obs, vx_obsp,yp_obs,vy_obsp,zp_obs,vz_obsp)




			#-------------External LOG------------------
			# Create a group to store velocity field for this iteration/time
			iteration_group = hf.create_group('LiDar_cyl_circle_results')
			# Have to log lidar readings in the actual lidar loop. this ie because when we append
			# the readings row after row, the rows will have different lengths. h5 tries to convert it
			# to numpy arrays which errors out because all the rows are differenet lengths. Se we have to log
			# the lidar readings iteratively/seperately
			# Psi Calculations - second quadrant obstacle
			iteration_group.create_dataset('x_cloverpe', data=x_pe)
			iteration_group.create_dataset('vx_cloverpe', data=vx_pe)
			iteration_group.create_dataset('y_cloverpe', data=y_pe)
			iteration_group.create_dataset('vy_cloverpe', data=vy_pe)
			iteration_group.create_dataset('z_cloverpe', data=z_pe)
			iteration_group.create_dataset('vz_cloverpe', data=vz_pe)
			iteration_group.create_dataset('psi_0p', data=psi_0p)
			iteration_group.create_dataset('psi_1p', data=psi_1p)
			iteration_group.create_dataset('r_cyl', data=r_cyl)
			iteration_group.create_dataset('time_now', data=timep)


			iteration_group = hf.create_group('LiDar_figure8_results')
			# Psi Calculations - cylinder first quadrant obstacle
			iteration_group.create_dataset('x_cloverce', data=x_ce)
			iteration_group.create_dataset('vx_cloverce', data=vx_ce)
			iteration_group.create_dataset('y_cloverce', data=y_ce)
			iteration_group.create_dataset('vy_cloverce', data=vy_ce)
			iteration_group.create_dataset('z_cloverce', data=z_ce)
			iteration_group.create_dataset('vz_cloverce', data=vz_ce)
			iteration_group.create_dataset('psi_0', data=psi_0)
			iteration_group.create_dataset('psi_1', data=psi_1)
			iteration_group.create_dataset('r_cyl', data=r_rec)
			iteration_group.create_dataset('time_now', data=timec)

			#-------------External LOG------------------
            # Create a group to store the control variables over the simulation
			iteration_group = hf.create_group('Control_log')
			iteration_group.create_dataset('x_clover', data=xf)
			iteration_group.create_dataset('y_clover', data=yf)
			iteration_group.create_dataset('z_clover', data=zf)
			iteration_group.create_dataset('yaw_clover', data=YawF)
			iteration_group.create_dataset('yaw_com', data=YawC)
			iteration_group.create_dataset('yaw_error', data=eyaw)
			iteration_group.create_dataset('velx_clover', data=velfx)
			iteration_group.create_dataset('vely_clover', data=velfy)
			iteration_group.create_dataset('velz_clover', data=velfz)
			iteration_group.create_dataset('velx_com', data=velcx)
			iteration_group.create_dataset('vely_com', data=velcy)
			iteration_group.create_dataset('velz_com', data=velcz)
			iteration_group.create_dataset('Ux_mpc', data=Ux)
			iteration_group.create_dataset('Uy_mpc', data=Uy)
			iteration_group.create_dataset('Uz_mpc', data=Uz)

			#time_now = np.array(time_now)
			iteration_group.create_dataset('time_now', data=time_now)

                #------------------------------------------
			

			# debug section
			# Plot logged data for analyses and debugging
			plt.figure(1)
			plt.subplot(211)
			plt.plot(xf,yf,zf,'r',label='pos-clover')
			plt.plot(xcom,ycom,zcom,'b',label='MPC-com')
			# plt.plot(self.posx,self.posy,'b--',label='x-com')
			# plt.fill(xa[0],ya[0],'k') # plot first reading
			plt.legend()
			plt.grid(True)
			#plt.subplot(312)
			#plt.plot(yf,'r',label='y-fol')
			#plt.plot(ya,'b--',label='y-obs')
			#plt.legend()
			#plt.grid(True)
			#plt.ylabel('Position [m]')
			plt.subplot(212)
			plt.plot(YawF,'b',label='yaw-F')
			plt.plot(YawC,'g',label='yaw-C')
			plt.legend()
			plt.grid(True)
			plt.ylabel('yaw [deg]')
			plt.xlabel('Time [s]')
			
			# Velocity plot
			plt.figure(2)
			plt.subplot(311)
			plt.plot(velfx,'r',label='vx-vel')
			plt.plot(velcx,'b',label='vx-MPC-com')
			plt.plot(uVPM,'g',label='vx-VPM')
			plt.ylabel('vel[m/s]')
			plt.xlabel('Time [s]')
			plt.legend()
			plt.grid(True)
			plt.subplot(312)
			plt.plot(velfy,'r',label='vy-vel')
			plt.plot(velcy,'b--',label='vy-MPC-com')
			plt.plot(vVPM,'g',label='vy-VPM')
			plt.legend()
			plt.grid(True)
			plt.ylabel('Position [m]')
			plt.subplot(313)
			plt.plot(velfz,'r',label='evx')
			plt.plot(velcz,'b',label='evy')
			plt.ylabel('vel[m/s]')
			plt.xlabel('Time [s]')
			plt.legend()
			plt.grid(True)

			plt.figure(3)
			plt.subplot(311)
			plt.plot(Ux,'r')
			plt.legend()
			plt.grid(True)
			plt.ylabel('Ux')
			plt.subplot(312)
			plt.plot(Uy,'b')
			plt.legend()
			plt.grid(True)
			plt.ylabel('Uy')
			plt.subplot(313)
			plt.plot(Uz,'r')
			plt.legend()
			plt.grid(True)
			plt.ylabel('T_input')
			plt.xlabel('Time [s]')
			

			fig = plt.figure(4)
			plt.cla()
			np.seterr(under="ignore")
			# plt.streamplot(x_field,y_field,u_field,v_field,linewidth=1.0,density=40,color='r',arrowstyle='-',start_points=XYsl) # density = 40
			# plt.streamplot(x_field,y_field,Vxe,Vye1,linewidth=1.0,density=40,color='r',arrowstyle='-',start_points=XYsl)
			plt.grid(True)
			#plt.plot(XX,YY,marker='o',color='blue')
			plt.axis('equal')
			plt.xlim(xVals)
			plt.ylim(yVals)
			plt.plot(xf,yf,'b',label='x-fol') # path taken by clover
			plt.xlabel('X Units')
			plt.ylabel('Y Units')
			plt.title('Streamlines with Stream Function Velocity Equations')

			plt.figure(5)
			plt.plot(psi_0,'r',label='psi_0')
			plt.plot(psi_1,'b',label='psi_1')
			plt.ylabel('b(x)')
			plt.xlabel('Time [s]')
			plt.legend()
			plt.grid(True)

			plt.figure(6)
			plt.plot(xc, yc, 'ro', label='ellipse-center', markersize=3)  # 'ro' for red circles
			plt.ylabel('center[m]')
			plt.xlabel('Time [s]')
			plt.legend()
			plt.grid(True)

			plt.figure(7)
			plt.plot(psi_0p,'r',label='psi_0p')
			plt.plot(psi_1p,'b',label='psi_1p')
			plt.ylim(-2, 30)
			plt.ylabel('b(x)')
			plt.xlabel('Time [s]')
			plt.legend()
			plt.grid(True)

			plt.figure(8)
			plt.plot(xcp, ycp, 'ro', label='ellipse-center', markersize=3)  # 'ro' for red circles
			plt.ylabel('center[m]')
			plt.xlabel('Time [s]')
			plt.legend()
			plt.grid(True)

			plt.show()
			
		except rospy.ROSInterruptException:
			pass

