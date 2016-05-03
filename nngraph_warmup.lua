require("nn")
require("nngraph")

function t_nngraph(x, y, z)
	local p1 = nn.Tanh()(nn.Linear(2, 5)(x))
	local p2 = nn.Sigmoid(nn.Linear(3, 5)(y))
	local p3 = nn.Square()(p1)
	local p4 = nn.Square()(p2)
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

function cons_net()
	local net = nn.Sequential()

end

graph_net = build_net()
x = torch.Tensor({2, 2}})
y = torch.Tensor({4, 4, 4})
z = torch.Tensor({5, 5, 5, 5, 5})

a = graph_net:forward({x, y, z})
print(a)

grad_output = torch.Tensor({1, 1, 1, 1, 1})
grad_input = graph_net:backward(grad_output)
print(grad_input)
