require("nn")
require("nngraph")

function t_nngraph(x, y, z)
	local p1 = nn.Tanh()(nn.Linear(2, 5)(x))
	local p2 = nn.Sigmoid(nn.Linear(3, 5)(y))
	local p3 = nn.CMulTable()({p1, p2})
	local p4 = nn.CMulTable()({p1, p2})
	local new_gate = nn.CMulTable()({p3, p4})
	local final = nn.CAddTable()({new_gate, z})
	return final
end

function build_net()
	local x = nn.Identity()()
	local y = nn.Identity()()
	local z = nn.Identity()()
	local output = t_nngraph(x, y, z)
	local module = nn.gModule({x, y, z}, {output})
	module:getParameters():uniform(-params.init_weight, params.init_weight)
	return module
end

graph_net = build_net()
x = torch.Tensor(2):random()
y = torch.Tensor(3):random()
z = torch.Tensor(5):random()

print(x)
print(y)
print(z)

print(unpack(graph_net:forward({x, y, z})))
