from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, norm_2
# Cadadi is used as a front end for Acados, Acados solver is much different and much faster/better than Casadi

import numpy as np

def export_clover_model():

	model_name = 'clover_model'
	# ccording to chatgpt, SX variable will be better for a problem where constraints are time verying and not static
	 # set up states & controls
	x = SX.sym('x')
	vx = SX.sym('vx')
	y = SX.sym('y')
	vy = SX.sym('vy')
	z = SX.sym('z')
	vz = SX.sym('vz')
	
	x = vertcat(x, vx, y, vy, z, vz) # State = [x, vx, y, vy, z, vz]
	
	ux = SX.sym('ux')
	uy = SX.sym('uy')
	uz = SX.sym('uz')

	u = vertcat(ux, uy, uz)
	
	# xdot
	ax = SX.sym('ax')
	ay = SX.sym('ay')
	az = SX.sym('az')
	
	xdot = vertcat(vx, ax, vy, ay, vz, az) # Derivative of the state wrt time
	
	
	# Explicit dynamics x_dot = f_expl(x,u,p)
	f_expl = vertcat(vx, ux, vy, uy, vz, uz)
	
	# implicit dynamics f_impl(xdot,x,u,z,p) = 0
	f_impl = xdot - f_expl

	# Set up the obstacle state parameters
	x_obs = SX.sym('x_obs')
	vx_obs = SX.sym('vx_obs')
	ax_obs = SX.sym('ax_obs')
	y_obs = SX.sym('y_obs')
	vy_obs = SX.sym('vy_obs')
	ay_obs = SX.sym('ay_obs')
	r_obs = SX.sym('r_obs')


	# Parameters (CasADi variable descibing parameters of the DAE; Default [].
	# parameters
	p = vertcat(x_obs, vx_obs, ax_obs, y_obs, vy_obs, ay_obs, r_obs) # Obstacle state and radius (major axis of ellipsoid estimation)
	
	
	# define model
	model = AcadosModel()
	
	model.f_impl_expr = f_impl
	model.f_expl_expr = f_expl
	model.x = x
	model.u = u
	model.xdot = xdot
	model.p = p
	model.name = model_name
	
	

	return model
