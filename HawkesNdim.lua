function HawkesNdim(alpha, beta, delta, lambda0, no_of_events,dim)

require 'nn'
require 'torch'


-- Initialize Output Tensors
local time 		= torch.zeros(no_of_events + 2,1)
local count 	= torch.zeros(no_of_events + 2, dim)
local lambda 	= torch.zeros(no_of_events + 2, dim)
local S 		= torch.zeros(dim)
local tag 		= torch.zeros(no_of_events)
for i = 1,dim do
	lambda[{1,i}] = lambda0[i]
end
-- loop through time step, t
for t = 2,no_of_events + 2 do
	for i = 1,dim do 
	-- Define D
	local U = torch.Tensor(1):uniform(0,1)
	local D =  1 + torch.log(U)*torch.Tensor(1):fill(delta[i])/(lambda[t-1][i] - alpha[i])

	-- Define potential S values
	
	-- Select S value from i = 1,dim and place in global S vector

	local S1  = (-1/delta[i])*torch.log(torch.Tensor(1):fill(D))
    local S2  = (-1/alpha[i])*torch.log(torch.Tensor(1):uniform(0,1))
   
    
	if D > 0 then
	S[i] = torch.cmin(S2,S1)
	
	else 
	S[i] = S2[1]
	end
end
	-- Select minimum S and set equal to W, and record associated index
	W, ind = torch.kthvalue(S,1)
	-- Record new jump time, t = (t-1) + W
	time[t] = time[t-1] + W 

	-- update across all dimensions
	for j = 1,dim do
	-- Copy forward event count, and increase the count with index S_min by 1
	count[t][j] = count[t-1][j]
		
	-- Update lambda values
	--
 	lambda[t][j] = (lambda[t-1][j] - alpha[j])*torch.exp(-delta[j]*(time[t] - time[t-1])) + alpha[j]
 	local lambda_minus	= torch.Tensor(1):exponential(beta[j][ind[1]])
 	lambda[t][j] = lambda[t][j] + lambda_minus[1]

 	end
 	count[t][ind[1]] = count[t][ind[1]] + 1

	-- Concatenate, create inputs/targets
	S:zero()
end
	
 	out = torch.cat(time,count)
 	out = torch.cat(out,lambda)

 	events = torch.zeros(no_of_events+2)

 	events[1] = time[1]
 	 	
 	for i = 2,no_of_events+2 do
 		events[i] = time[i] - time[i-1]
 	end
 	
 	inputs = events:sub(2,no_of_events+1)
 	targets = events:sub(3, no_of_events + 2)

 	return inputs,targets

end