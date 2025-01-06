import torch
import torch.nn.functional as F


class DecorLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        decorrelation_method: str = "copi",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert in_features == out_features, "DecorLinear only supports square matrices"
        assert decorrelation_method in [
            "copi",
        ], "DecorLinear only supports 'copi'"
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.decorrelation_method = decorrelation_method

        self.weight = torch.nn.Parameter(
            torch.empty(in_features, in_features, **factory_kwargs),
        )

        self.bias = None
        if bias is not None:
            self.bias = torch.nn.Parameter(
                torch.empty(in_features, **factory_kwargs),
            )

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.eye = torch.nn.Parameter(
            torch.eye(in_features, **factory_kwargs),
        )

        self.reset_decor_parameters()

    def reset_decor_parameters(self):
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        torch.nn.init.eye_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.decorrelated_state = F.linear(input, self.weight, self.bias)
        return self.decorrelated_state

    def update_grads(self, _) -> None:
        assert self.decorrelated_state is not None, "Call forward() first"

        # If using a bias, it should demean the data
        if self.bias is not None:
            self.bias.grad = self.decorrelated_state.sum(axis=0)

        corr = (1 / len(self.decorrelated_state)) * (
            self.decorrelated_state.transpose(0, 1) @ self.decorrelated_state
        )

        # The off-diagonal correlation = (1/batch_size)*(x.T @ x)*(1.0 - I)
        off_diag_corr = corr * (1.0 - self.eye)

        # gradient calc
        grads = off_diag_corr @ self.weight

        # Zero-ing the decorrelated state so that it cannot be re-used
        self.decorrelated_input = None

        # Update grads of decorrelation matrix
        self.weight.grad = grads

    def get_fwd_params(self):
        return []

    def get_decor_params(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class HalfBatchDecorLinear(DecorLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        half_batch_width = len(input) // 2

        self.undecorrelated_state = input.clone().detach()[:half_batch_width]
        self.decorrelated_state = output.clone().detach()[:half_batch_width]
        return output
