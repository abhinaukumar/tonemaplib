import torch

class tmqi(torch.autograd.Function):
	"""
        The implementation of the tmqi cost function: SSIM and Naturalness
        """
        @staticmethod
	def forward(obj, input):
		obj.save_for_backward(input)
                
                Pm = K*exp((-m+115.94)/1566.88)
                Pd = np.power(1-d,9)*np.power(d,3.4)/1.9961e-04
                if Pm > Pd:
                    N = Pd
                else:
                    N = Pm
                Tmqi = .082*S.^(0.3046) + .912*N.^(0.7088)
		return input.clamp(min=0)


	@staticmethod
	def backward(obj, grad_output):
		input, =obj.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input<0] = 0 
		return grad_input


