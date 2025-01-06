import torch


class NPLinearFunc(torch.autograd.Function):
    """Linear layer with noise injection at nodes"""

    @staticmethod
    def forward(ctx, input, weights, biases, sigma, clean_pass, dist_sampler):
        # Matrix multiplying both the clean and noisy forward signals
        output = torch.mm(input, weights.t())

        if biases is not None:
            output += biases

        # Determining the shape of the noise
        noise_shape = [s for s in output.shape]
        noise_shape[0] = noise_shape[0] // 2  # int division

        if clean_pass:
            noise_1 = torch.zeros(noise_shape).to(input.device)
        else:
            noise_1 = sigma * dist_sampler(noise_shape).to(input.device)
        noise_2 = sigma * dist_sampler(noise_shape).to(input.device)

        # Generating the noise
        noise = torch.concat([noise_1, noise_2])

        # Adding the noise to the output
        output += noise

        # compute the output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None


class NPLinear(torch.nn.Linear):
    """Node Perturbation layer with saved noise"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float,
        clean_pass: bool,
        dist_sampler: torch.distributions.Distribution,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.sigma = sigma
        self.clean_pass = clean_pass
        self.dist_sampler = dist_sampler

    def __str__(self):
        return "NPLinear"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # A clean and noisy input are both processed by a layer to produce
        output = NPLinearFunc().apply(
            input,
            self.weight,
            self.bias,
            self.sigma,
            self.clean_pass,
            self.dist_sampler,
        )
        half_batch_width = len(input) // 2

        self.clean_input = input[:half_batch_width]
        self.output_diff = output[:half_batch_width] - output[half_batch_width:]
        self.square_norm = torch.sum(
            self.output_diff.reshape(half_batch_width, -1) ** 2, axis=1
        )
        return output

    def update_grads(self, scaling_factor) -> None:
        with torch.no_grad():
            # Rescale grad data - to be used at end of gradient pass
            scaled_out_diff = scaling_factor[:, None] * self.output_diff
            self.weight.grad = torch.einsum(
                "ni,nj->ij", scaled_out_diff, self.clean_input
            )

            if self.bias is not None:
                self.bias.grad = torch.einsum("ni->i", scaled_out_diff)

    def get_noise_squarednorm(self):
        return self.square_norm

    def get_number_perturbed_params(self):
        return self.out_features

    def get_fwd_params(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def get_decor_params(self):
        return []
