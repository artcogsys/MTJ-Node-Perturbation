from . import decor
from . import np
from . import bp
from . import noise
import torch
import numpy


def construct_net(in_size, out_size, layer_type, config):
    if config.learning_rule == "BP":
        model = NoPerturbNet(
            in_size=in_size,
            out_size=out_size,
            n_hidden_layers=config.network.n_hidden_layers,
            hidden_size=config.network.hidden_layer_size,
            layer_type=layer_type,
            decorrelation=config.optimizer.decor_lr != 0.0,
            decorrelation_method=config.optimizer.decor_method,
            biases=config.network.bias,
        )
    else:
        model = PerturbNet(
            in_size=in_size,
            out_size=out_size,
            n_hidden_layers=config.network.n_hidden_layers,
            hidden_size=config.network.hidden_layer_size,
            sigma=config.noise.sigma,
            clean_pass=config.noise.clean_pass,
            noise_distribution=config.noise.distribution,
            noise_config=config.noise,
            layer_type=layer_type,
            decorrelation=config.optimizer.decor_lr != 0.0,
            decorrelation_method=config.optimizer.decor_method,
            biases=config.network.bias,
            device=config.device,
        )
    return model


class PerturbNet(torch.nn.Module):
    def __init__(
        self,
        in_size=28 * 28,
        out_size=10,
        hidden_size=1000,
        n_hidden_layers=0,
        sigma=1e-5,
        clean_pass=True,
        noise_distribution="normal",
        noise_config=None,
        layer_type=np.NPLinear,
        activation_function=torch.nn.LeakyReLU(),
        biases=True,
        decorrelation=False,
        decorrelation_method="copi",
        device="cpu",
    ):
        super(PerturbNet, self).__init__()
        self.layers = []
        self.decorrelation = decorrelation
        self.N = hidden_size * n_hidden_layers + out_size

        if noise_distribution == "Normal":
            distribution = torch.distributions.Normal(
                torch.tensor([noise_config.offset + noise_config.p_skew])
                .to(torch.float32)
                .to(device),
                torch.tensor([1.0]).to(device),
            )
            dist_sampler = (
                lambda x: distribution.sample([noise_config.n_sources] + x)
                .squeeze_(-1)
                .sum(0)
            )
        elif noise_distribution == "Bernoulli":
            distribution = torch.distributions.Bernoulli(
                torch.tensor([0.5 + noise_config.p_skew]).to(torch.float32).to(device)
            )
            dist_sampler = (
                lambda x: noise_config.offset * noise_config.n_sources
                + distribution.sample([noise_config.n_sources] + x).squeeze_(-1).sum(0)
            )
        elif noise_distribution == "CentredBernoulli":
            distribution = torch.distributions.Bernoulli(
                torch.tensor([0.5 + noise_config.p_skew]).to(torch.float32).to(device)
            )
            dist_sampler = lambda x: distribution.sample(
                [noise_config.n_sources] + x
            ).squeeze_(-1).sum(0) - noise_config.n_sources * (0.5 + noise_config.p_skew)
        elif noise_distribution == "sMTJ":
            distribution = noise.sMTJNoise(
                noise_config.path, torch.tensor([noise_config.sigma]).to(device)
            )
            dist_sampler = lambda x: distribution.sample([noise_config.n_sources] + x)
        elif noise_distribution == "simulatedMTJ":
            pass  # If simulatedMTJ then we need to know number of nodes per layer

        for i in range(n_hidden_layers + 1):
            in_dim = in_size if i == 0 else hidden_size
            out_dim = hidden_size if i < n_hidden_layers else out_size
            if decorrelation:
                self.layers.append(
                    decor.HalfBatchDecorLinear(
                        in_dim,
                        in_dim,
                        bias=False,
                        decorrelation_method=decorrelation_method,
                    )
                )

            if noise_distribution == "simulatedMTJ":
                dist_sampler = noise.simulatedMTJNoise(
                    torch.tensor([noise_config.sigma]).to(device),
                    n_sources=noise_config.n_sources,
                    num_nodes=out_dim,
                )

            self.layers.append(
                layer_type(
                    in_dim,
                    out_dim,
                    sigma,
                    clean_pass,
                    dist_sampler,
                    bias=biases,
                )
            )

        self.activation_function = activation_function
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for indx, layer in enumerate(self.layers):
            x = layer(x)
            if (indx + 1) < len(self.layers) and not isinstance(
                layer, decor.HalfBatchDecorLinear
            ):
                x = self.activation_function(x)
        return x

    def train_step(self, data, target, onehots, loss_func):
        with torch.inference_mode():
            # Duplicate data for network clean/noisy pass
            output = self(torch.concatenate([data, data.clone()]))
            normalization = self.get_total_normalization()
            clean_loss = loss_func(
                output[: len(data)], target, onehots
            )  # sum up batch loss
            noisy_loss = loss_func(
                output[len(data) :], target, onehots
            )  # sum up batch loss
            # clean_loss.sum().backward()
            # Multiply grad by loss differential and normalize with unit norms
            loss_differential = clean_loss - noisy_loss
            multiplication = loss_differential * normalization
            for layer in self.layers:
                layer.update_grads(multiplication)

            return clean_loss.sum()

    def test_step(self, data, target, onehots, loss_func):
        with torch.inference_mode():
            output = self(torch.concatenate([data, data.clone()]))
            loss = torch.sum(
                loss_func(output[: len(data)], target, onehots)
            ).item()  # sum up batch loss
            return loss, output[: len(data)]

    def get_fwd_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_fwd_params())
        return params

    def get_decor_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_decor_params())
        return params

    def get_total_normalization(self):
        normalization = 0.0
        num_perturbed_params = 0
        for layer in self.layers:
            if hasattr(layer, "get_noise_squarednorm"):
                normalization += layer.get_noise_squarednorm()
                num_perturbed_params += layer.get_number_perturbed_params()
        return num_perturbed_params / normalization


class NoPerturbNet(torch.nn.Module):
    def __init__(
        self,
        in_size=28 * 28,
        out_size=10,
        hidden_size=1000,
        n_hidden_layers=0,
        layer_type=bp.BPLinear,
        activation_function=torch.nn.LeakyReLU(),
        biases=True,
        decorrelation=False,
        decorrelation_method="copi",
    ):
        super(NoPerturbNet, self).__init__()
        self.layers = []
        self.decorrelation = decorrelation

        for i in range(n_hidden_layers + 1):
            in_dim = in_size if i == 0 else hidden_size
            out_dim = hidden_size if i < n_hidden_layers else out_size
            if decorrelation:
                self.layers.append(
                    decor.DecorLinear(
                        in_dim,
                        in_dim,
                        bias=False,
                        decorrelation_method=decorrelation_method,
                    )
                )
            self.layers.append(
                layer_type(
                    in_dim,
                    out_dim,
                    bias=biases,
                )
            )

        self.activation_function = activation_function
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for indx, layer in enumerate(self.layers):
            x = layer(x)
            # TODO: Don't activation func if decor
            if (indx + 1) < len(self.layers) and not isinstance(
                layer, decor.DecorLinear
            ):
                x = self.activation_function(x)
        return x

    def train_step(self, data, target, onehots, loss_func):
        # Duplicate data for network clean/noisy pass
        output = self(data)
        loss = loss_func(output[: len(data)], target, onehots)
        total_loss = loss.sum()
        total_loss.backward()
        for layer in self.layers:
            if isinstance(layer, decor.DecorLinear):
                layer.update_grads(None)
        return total_loss

    def test_step(self, data, target, onehots, loss_func):
        with torch.inference_mode():
            output = self(data)
            loss = torch.sum(
                loss_func(output, target, onehots)
            ).item()  # sum up batch loss
            return loss, output

    def get_fwd_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_fwd_params())
        return params

    def get_decor_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_decor_params())
        return params
