--run.lua

require 'train_1dim_hawkes'

hdim 			= { [2] = 100, [1] = 100}
dt0 			= 1.5
step_reduction 	= .75

train_1dim_hawkes(hdim, dt0, step_reduction)