import torch
x = torch.tensor([1.0,2.0,3.0], requires_grad=True)
y = torch.tensor([3.0,2.0,1.0], requires_grad=True)
z = (x*y).sum().pow(2)
z.backward(create_graph=True)
gradx = x.grad
partial_xy = [torch.autograd.grad(gradx[i], y, retain_graph=True, only_inputs=True)[0] 
			for i in range(gradx.numel())]
partial_xy = torch.stack(tuple(partial_xw), 0)