from CLOVER_MODEL import export_clover_model
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import vertcat, sum1, mtimes, Function, norm_1, norm_2, dot
import numpy as np
import scipy.linalg



def acados_settings(N_horizon, T_horizon):

    # optimal control problem
        # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcp
		ocp = AcadosOcp()

        # https://docs.acados.org/python_interface/index.html#acados_template.acados_model.AcadosModel
		model = export_clover_model()
		ocp.model = model
		x = model.x
		u = model.u
		# Dimensions
		nx = model.x.size()[0]  # number of states.
		nu = model.u.size()[0]  # number of inputs.
		ny = nx + nu # y is x and u concatenated for compactness of the loss function
		ny_e = nx # u is not a decision variable for the final in the prediction horizon

        # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpDims
		ocp.dims.N = N_horizon # prediction horizon, other fields in dim are set automatically


        # Set the Cost (built in cost function):
		ocp.cost.cost_type = "LINEAR_LS"  # NONLINEAR_LS  --> Cost type at intermediate shooting nodes (1 to N-1): Default LINEAR_LS
        # ocp.cost.cost_type_0 = "NONLINEAR_LS" # Cost type at initial shooting node (0): would be the same as intermediate shooting nodes if not set explicitly
        # # Therefore did not need to set the type_0 if it was the same
		ocp.cost.cost_type_e = "LINEAR_LS" # Cost type at terminal shooting node (N): Default LINEAR_LS
		# Optimization costs
		ocp.cost.Vx = np.zeros((ny, nx)) # raise the dim of x to the dim of y
		ocp.cost.Vx[:nx, :nx] = np.eye(nx)  # Weight only x (zeros in Q will remove some of these)
		ocp.cost.Vx_e = np.eye(nx) # enx x cost

		ocp.cost.Vu = np.zeros((ny, nu)) # raise dim of u to the dim of y
		ocp.cost.Vu[-nu:, -nu:] = np.eye(nu) # weight only u

		# Cost matrices
		# Q = np.array([ 10, 0, 10, 0, 10, 0]) # Assuming there are only 3 state outputs, State = [x, vx, y, vy, z, vz]
		# R = np.array([ 1e-4, 1e-4, 1e-4]) # Three control inputs acceleration
		#Q = np.array([ 1000, 0, 1500, 0, 1500, 0]) # Assuming there are only 3 state outputs, State = [x, vx, y, vy, z, vz]
		Q = np.array([ 0, 100, 0, 100, 50, 30]) # 50-80 80-100 in position 40-50 on velocity
		Q_e = np.array([ 0, 100, 0, 100, 50, 30])
		R = np.array([ 5, 5, 5]) # 5 is good
		

		ocp.cost.W_e = np.diag(Q_e)  # inputs are not decision variables at the end of prediction horizon
		ocp.cost.W = np.diag(np.concatenate((Q,R))) # acados combines states and control input to yref for compactness

        # Set the cost (Custom cost function):
        # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpCost
        # self.ocp.cost.cost_type = "EXTERNAL"
        # self.ocp.cost.cost_type_e = "EXTERNAL"

        # Dont need to explicitly define these as they are defulated to 'casadi':
        # ocp.cost.cost_ext_fun_type = 'casadi'
        # ocp.cost.cost_ext_fun_type_e = 'casadi'

        # Define cost/weight matrices
        # Q_c = 
        # self.ocp.cost_expr_ext_cost =  CloverCost(self.ocp)  # Experssion for external cost
        # ocp.cost_expr_ext_cost_e =  CloverCost_e(ocp)  # Terminal shooting point cost


        # ocp.cost_expr_ext_cost =  CloverCost(ocp)  # Experssion for external cost
        # ocp.cost_expr_ext_cost_e =  CloverCost_e(ocp)  # Terminal shooting point cost

		# Set the reference trajectory (this will be overwritten later, just for dimensions right now)
		x_ref = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
		ocp.cost.yref = np.concatenate((x_ref,np.array([0.0, 0.0, 0.0]))) # wont matter what we apply for refs for states with 0 weighting in Q above
		# yref is of length ny, and we are sendinging zeros for yref corresponding to control inputs ux,uy,uz
		ocp.cost.yref_e = np.zeros((ny_e,)) # end node reference (doesnt include control inputs of course)

		ocp.constraints.x0 = x_ref # initial state (not sure) translated internally to idxbx_0, lbx_0, ubx_0, idxbxe_0

		# Initialize the state of the target obstacle (this will be overwritten later, just initializing it right now) -> sets the model.p values define in model defintion
		ocp.parameter_values = np.array([0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0]) # State = [x, vx, ax, y, vy, ay, robs]

        # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpConstraints
        # constraints u = [ux,uy,uz] -> acceleration commands
		u_lb = np.array([-100, -100, -100])
		u_ub = np.array([100, 100, 100])

		ocp.constraints.constr_type = 'BGH'  # BGP is for convex over nonlinear.
		ocp.constraints.lbu = u_lb
		ocp.constraints.ubu = u_ub
		ocp.constraints.idxbu = np.array([0, 1, 2]) # Constraints apply to u[0],u[1],u[2]

        # Nonlinear inequality constraints using CBF
		# gamma = 0.1
		# x_obs = 2
		# y_obs = 2
		# r_obs = 1.0
		# ocp.model.con_h_expr = 2*((model.x[0] - x_obs)*model.x[7] + (model.x[1]-y_obs)*model.x[8])  - gamma*((model.x[0]-x_obs)**2 + (model.x[1]-y_obs)**2 - r_obs**2)
		# # ocp.model.con_h_expr = ((model.x[0]-x_obs)**2 + (model.x[1]-y_obs)**2 - r_obs**2) # Hard constraint avoidance
		# h_lb = np.array([0])
		# h_ub = np.array([10000])
		# ocp.constraints.lh = h_lb
		# ocp.constraints.uh = h_ub
		# # Usage of slack variables to relax the above hard constraints
		# ocp.constraints.Jsh = np.eye(1)
		# # slacks
		# L2_pen = 1e3 # 1e3
		# L1_pen = 1  #1
		# Nonlinear inequality constraints using CBF
		# State = [x, vx,ax, y, vy, ay]
		# Obstacle state
		state = ocp.parameter_values
		x_obs = state[0]
		vx_obs = state[1]
		ax_obs = state[2]
		y_obs = state[3]
		vy_obs = state[4]
		ay_obs = state[5]
		r_obs = state[6]

		# it is not possible to add and remove constraints in acados after creating the solver.
		# This is because the allocated memory is dependent on the dimensions of the problem, part of which are the number of constraints at each shooting node.

		# Radius of obstacle to avoid
		r = model.p[6] # 1.2
		# TODO Update obstacle states here
		# State = x_obs, vx_obs, ax_obs, y_obs, vy_obs, ay_obs
		q1 = 15#15
		q2 = 10#10
		delta_p = vertcat(model.x[0]-model.p[0], model.x[2] - model.p[3])
		delta_v = vertcat(model.x[1] - model.p[1], model.x[3] - model.p[4])
		delta_a = vertcat(model.u[0]-model.p[2], model.u[1] - model.p[5])



		norm_delta_p = norm_2(delta_p)#np.linalg.norm(delta_p, ord=1)
		norm_delta_v = norm_2(delta_v)#np.linalg.norm(delta_v, ord=1)

		c_ol = (norm_delta_v**2)/norm_delta_p - ((dot(delta_p,delta_v))**2)/(norm_delta_p**3) + (q1+q2)*(dot(delta_p,delta_v))/norm_delta_p + q1*q2*(norm_delta_p - r)

		ocp.model.con_h_expr = c_ol + dot(delta_p,delta_a)/norm_delta_p

		# ocp.model.con_h_expr_e = c_ol + np.dot(delta_p,delta_a)/norm_delta_p

		h_lb = np.array([0])
		h_ub = np.array([100000])
		ocp.constraints.lh = h_lb
		ocp.constraints.uh = h_ub

		# Usage of slack variables to relax the above hard constraints
		ocp.constraints.Jsh = np.eye(1)

		# # slacks
		L2_pen = 1e5 # 1e4
		L1_pen = 1e4  #1e3

		ocp.cost.Zl = L2_pen*np.ones((1,)) # Diagonal of hessian WRT lower slack
		ocp.cost.Zu = L2_pen*np.ones((1,))
		ocp.cost.zl = L1_pen*np.ones((1,)) # Gradient with respect to lower slack at intermediate shooting nodes
		ocp.cost.zu = L1_pen*np.ones((1,))

		# ocp.constraints.lh_e = h_lb
		# ocp.constraints.uh_e = h_ub

		# Gradient indicates how changes in slack variables influence cost function. Higher gradient value implies stronger influence of the slack variables
		# on the cost. It determines rate of change of the cost wrt to changes in alsack variables
		# Hessians wrt slack variables measures the curvature of the cost function wrt to changes in slack variables, diagonals of hessian represent
		# the second derivatives of the cost function wrt corresponding slack variables
		# Higher hessian value indicates how rapidly the gradient of the cost function changes wrt slack variables

		# Higher gradient and higher hessian increase penalty for violations of the lower or upper bounds of slack variables


        # Solver options
        # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp_options.AcadosOcpOptions
    
        # solver_options.Tsim is time horizon for one integrator step. Automatically as tf/N. Deafault: None
		ocp.solver_options.tf = T_horizon
		ocp.solver_options.integrator_type = 'ERK' # ERK explicit numerical integration based on Runge-Kutta scheme, suitable for simple dynamics and fast sampling times
        # IRK implicit numerical integration based on runge-kutta scheme, suitable for complex dynamics
		ocp.solver_options.nlp_solver_type = 'SQP_RTI' # 'SQP_RTI'
        # SQP - sequential quadratic programming method, simple linear problems
        # SQP_RTI extension of SQP to real time applications, aims to reduce computational time
		ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES

		 # Create solvers
        
		acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json")
		acados_integrator = AcadosSimSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json")

		return acados_solver, acados_integrator, model

def CloverCost(ocp):

	# Extract Parameters
	x = ocp.x  # States are N*n in dimensions where N are the shooting nodes and n are the states
	u = ocp.u
	N = ocp.dims.N  # Prediction horizon i.e. number of shooting nodes
	
	ref = ocp.cost.yref
	
	# State and input weight matrices
	
	Q = vertcat(10,10,10) # weight matrix for the tracking error
	R = vertcat(R_ele, R_ele, R_ele)  # Weight matrix for control input
	R_du = vertcat(0.0001, 0.0001, 0.0001) # Weight matrix for input smoothness
	
	# Extract the current state (not a decision variable) x,y positions
	X_cur = x[0, [0,2]]
	
	# Remember from matlab,
	X_dec = x[1:N+1, [0, 2, 4]] # Extract position [x, y, z] from 1->N shooting nodes
	U_dec1 = u[0:N, :] #--> goes from positions 0->N-1 shooting nodes

	U_dec2 = u[1:N, :] # Goes from 1->N-1
	U_dec2 = vertcat([0, 0, 0], U_dec2)

	# Error variables
	E_X = ref[1:N+1, :] - X_dec.T # Columns = outputs, rows = prediction horizon i.e. shooting nodes
	Delta_u = U_dec2 - U_dec1 # Change in u per step in prediction horizon, for smoother control input

	# Collision avoidance potential function
	r_des = 2.2

	# Avoidance via distance from center of circle
	X_crit = x_center*np.ones((N,1)) # Size Nx1 array
	Y_crit = y_center*np.ones((N,1))
	XY_crit = vertcat(X_crit, Y_crit)

	R_des = r_des*np.ones((N,1))

	E_Xc = XY_crit - X_dec[:, 0:2] # Uncludes column 0 and up to, but not including 
									# = X_dec[:, [0, 1]]
	mag = sum1(E_Xc**2, axis=1) # sum1 from casadi
	d_mag = R_des**2 - mag

	# potential function parameters:
	lambda_val = 1500
	k = 5
	Joa = lambda_val/(1 + exp(-k*d_mag))

	# Compute the cost function
	J = 0

	for i in range(N):
		Je = mtimes([E_X[:,i].T, Q, E_X[:,i]]) # Tracking error cost component
		Ju = mtimes([U_dec1[:,i].T, R, U_dec1[:, 1]]) # Control input cost component
		Jdu = mtimes([Delta_u[:,i].T, R_du, Delta_u[:,i]]) # Smoothness of control input cost
		Jep = 0.0001*e # Slack variable to alleviate strickness

		J1 = Je + Ju + Jdu # Joa[i] # Include the relevant contributions
		J += J1

	# Optionally, include soft constraints penelty with slack (e)
	#J += 0.0001*e


	



	# Create a CasADi Function for the cost expression
	J_function = Function('CloverCost', [x,u], [J])

	return J_function
	
	
	
