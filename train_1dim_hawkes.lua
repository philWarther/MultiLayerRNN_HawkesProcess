function train_1dim_hawkes(hdim, dt0, step_reduction)

--[[
A self-exciting event is something where one occurence of an event is more likely 
to trigger repeated occurences of that same event. For example, an event like an earthquake is likely 
to trigger succesive earthquake (aftershocks). Events of this nature are known as a Hawkes Process

This function trains recurrent neural network (RNN) to learn the elasped time between event 
occurences, known as inter-arrival times, in a Hawkes Process using statiscially 
simulated data for training.

Validation for the network is computed by comparing the error of the predicted inter-arrival times
to the average inter-arrival. The relative error (or error relative to the average) can be seen 
as a percentage improvement over using the average inter-arrival time to predict event occurences

INPUTS

dim -- 				A table denoting the dimensions of the weights for each layer of,
					the network. The number of entries in the table (#dim) will determine
					the number of layers in the network

dt0 --				The initial learning rate for the gradient descent algorithm

step_reduction -- 	When learning shows no improvment over several epochs, the learning rate is 
					mulitplied by this number to improve learning
					note: The stopping criteria of the training process is triggered once the 
					learning rate is reduced in size to a certain threshold
]]

require 'nn'
require 'nngraph'
require 'xlua'
require 'HawkesNdim'
require 'build_module_Nlayer'
require 'getmodule'
require 'pause'


-- Parameters for sequence generation

dim 			= 1
epsilon 		= 10^(-8)
alpha 			= torch.Tensor(dim):fill(1.5)--uniform(.5,5)
beta 			= torch.ones(dim,dim):fill(0.75)--uniform(.5,8)
delta 			= torch.ones(dim):fill(2.0)--uniform(.1,5)
lambda0 		= torch.ones(dim):fill(2.0)--uniform(alpha[1] + epsilon, alpha[1] + 5)

-- Initializing Tables
model 				= {}
model_table 	 	= {}

xdim				= 1			-- Dimension of the input vector, x
event_count			= 100			-- Length of the data sequence
criterion 			= nn.MSECriterion() -- Mean Squared Error Criterion Function
criterion.sizeAverage = false


local function replicate_module()
	mx,mdx = model:parameters()
	for i = 1,#mx do
		mx[i]:uniform(-0.07, 0.07)
	end

	local addf1 = getmodule(model,'f1')
	for i = 1,#addf1 do
		addf1[i].bias:fill( 1.0 )
	end

	local addf2 = getmodule(model,'f2')
	for i = 1, #addf2 do
		addf2[i].bias:fill(-1.0)
	end

	for t = 1,event_count do
		model_table[t]=model:clone()
		px,pdx = model_table[t]:parameters()

		for i = 1,#px do
			px[i]:set(mx[i])
			pdx[i]:set(mdx[i])
		end
	end
	collectgarbage()
end

local function forward_pass()
	local model_error 	= 0
	local average_error = 0
	h_in[1] 		= torch.Tensor(1):fill(inputs[1])
	local y_out 	= model_table[1]:forward(h_in)
	local crit_out 	= criterion:forward(y_out[1],torch.Tensor(1):fill(targets[1]))
	local avg_out 	= criterion:forward(torch.Tensor(1):fill(average),torch.Tensor(1):fill(targets[1]))

	model_error 	= model_error + crit_out
	average_error 	= average_error + avg_out

	for t= 2, event_count do
		h_out[1] = torch.Tensor(1):fill(inputs[t])
		for l = 2,#hdim+1 do
			h_out[l]:copy(model_table[t-1].output[l])
		end
		local y_out 	= model_table[t]:forward(h_out)
		local crit_out	= criterion:forward(y_out[1],torch.Tensor(1):fill(targets[t]))
		local avg_out 	= criterion:forward(torch.Tensor(1):fill(average),torch.Tensor(1):fill(targets[t]))

		model_error 	= model_error + crit_out
		average_error 	= average_error + avg_out

	end
	for l = 2,#hdim+1 do
			h_out[l]:copy(model_table[event_count].output[l])
	end
	return model_error,average_error 
end

local function backward_pass()
	local dG = {}

	for i = 2,#hdim + 1 do 
	 	dG[i] 		= torch.zeros(hdim[i-1])
	end
	for t = event_count,2,-1 do
		dG[1]	= criterion:backward(model_table[t].output[1], torch.Tensor(1):fill(targets[t]))
		dG 		= model_table[t]:backward(model_table[t-1].output,dG)
	end	
	dG[1] 		= criterion:backward(model_table[1].output[1], torch.Tensor(1):fill(targets[1]))
	model_table[1]:backward(h0,dG)
end

function HawkesAvg()
	local events 	= 100000
	local inp,out  	= HawkesNdim(alpha, beta, delta, lambda0, events, dim)
	local average 	= torch.mean(out)
	return average
end


local function iterate()
	local model_error 	= 0
	local average_error	= 0
	local dt 	= dt0--/event_count
		for j=1,#mdx do
			mdx[j]:zero()
		end
		x_loss,avg_loss = forward_pass()
		backward_pass()
		for j = 1,#mx do
			local Grad = 1.0/(torch.sqrt(#mx)*mdx[j]:norm())
			mx[j]:add(-dt*Grad*mdx[j])
		end

		model_error 	= model_error + x_loss
		average_error 	= average_error + avg_loss

		local relative_error 	= model_error/average_error

	return 	relative_error, model_error
end 

model 		= build_module_Nlayer(hdim,xdim)
replicate_module()
average 	= HawkesAvg()
history_new	= torch.zeros(50)
history_old = torch.zeros(50)

local relative_error
local average_error

-- Initializing Hidden State vectors
h0 			= {}
h_out		= {}
h_in		= {}
h0[1] 		= torch.zeros(xdim)
h_out[1]	= torch.zeros(xdim)
h_in[1]		= torch.zeros(xdim)


for i = 2,#hdim+1 do
	h0[i] 		= torch.zeros(hdim[i-1])
	h_in[i] 	= torch.zeros(hdim[i-1])
	h_out[i] 	= torch.zeros(hdim[i-1])
end

i = 0
while dt0>.001 do
	i = i + 1
	inputs,targets = HawkesNdim(alpha, beta, delta, lambda0, event_count,dim)
	relative_error, model_error = iterate()
	history_new[i%50+1] = model_error

	if i%50 == 0 then 
		new = torch.median(history_new)
		old = torch.median(history_old)
		history_old:copy(history_new)
		if new[1]>.999*old[1] and i>51 then
			dt0 = step_reduction*dt0
		print(' ---------- Step size updated to ' .. dt0 .. '------------')
		end
		print('Error Relative to the Average =  ' .. relative_error .. ' \tAbsolute Error = ' .. model_error)
	end

end 
local final = torch.mean(history_old)
print('\n---------- Final Results ----------')
print('Total Epochs \t\t\t= ' .. i .. '  \nError Relative to the Average \t=  ' .. relative_error .. ' \nAbsolute error \t\t\t= ' .. model_error )
return relative_error, model_error, final 

end