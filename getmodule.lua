function getmodule(model,tag)

	local thismod = {}
	for i,node in ipairs(model.backwardnodes) do

		if node.data.annotations.name == tag then
			table.insert(thismod,node.data.module)
		end

	end

	return thismod

end