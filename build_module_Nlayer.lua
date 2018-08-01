function build_module_Nlayer(hdim,xdim)
	local h 			= {}
	local h_out			= {}

	for i =1,#hdim+1 do
		h[i] 			= nn.Identity()()
	end
	

	for i = 2,#hdim + 1 do
		local Wx
		local W_f1_x
		local W_f2_x


		if i == 2 then
			Wx 		= nn.Linear(xdim,hdim[1])(h[1]) 
			W_f1_x 	= nn.Linear(xdim,hdim[1])(h[1]) 
			W_f2_x 	= nn.Linear(xdim,hdim[1])(h[1])	
		else
			Wx 		= nn.Linear(hdim[i-2],hdim[i-1])(h_out[i-1])
			W_f1_x 	= nn.Linear(hdim[i-2],hdim[i-1])(h_out[i-1])
			W_f2_x 	= nn.Linear(hdim[i-2],hdim[i-1])(h_out[i-1])
		end

		local U_f1_h 		= nn.Linear(hdim[i-1],hdim[i-1],false)(h[i])
		local U_f2_h 		= nn.Linear(hdim[i-1],hdim[i-1],false)(h[i])
		local f1 			= nn.Add(hdim[i-1])(nn.CAddTable()({U_f1_h,W_f1_x})):annotate{name = 'f1'}
		local f2 			= nn.Add(hdim[i-1])(nn.CAddTable()({U_f2_h,W_f2_x})):annotate{name = 'f2'}
		local sig_f1		= nn.Sigmoid()(f1)
		local sig_f2		= nn.Sigmoid()(f2)
		local Wx_tanh		= nn.Tanh()(Wx) 
		local h0_tanh		= nn.Tanh()(h[i])
		h_out[i]			= nn.CAddTable()({nn.CMulTable()({sig_f1,h0_tanh}),nn.CMulTable()({sig_f2,Wx_tanh})})
	end

	--h_out[1]		 		= nn.LogSoftMax()(nn.Linear(hdim[#hdim], cl)(h_out[#hdim+1])) -- do I want the linear?
	h_out[1]		 		= nn.Linear(hdim[#hdim], xdim)(h_out[#hdim+1]) -- do I want the linear?

	local model 			= nn.gModule(h,h_out)
	return model
end