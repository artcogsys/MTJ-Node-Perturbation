import torch
import math
import pandas as pd
import os


class sMTJNoise(torch.distributions.distribution.Distribution):
    """sMTJ Noise Distribution"""

    def __init__(self, path, scaling_factor):
        print("Using sMTJ Noise Distribution ...")
        self.scaling_factor = scaling_factor
        self.device = scaling_factor.device
        self.load_data(path)
        super(sMTJNoise, self).__init__(validate_args=False)

    def load_data(self, path):
        # Get all chunk filenames
        filenames = [name[:-4] for name in os.listdir(path) if name[-4:] == ".pqt"]
        n_chunks = len(filenames)

        # get shortest string from filenames
        filename = min(filenames, key=len)[:-1]

        # Load data chunks
        data = [pd.read_parquet(f"{path}/{filename}{i}.pqt") for i in range(n_chunks)]

        # Get total length of data across all chunks
        total_length = sum([len(chunk) for chunk in data])

        # Load all of the data to device
        self.data = torch.empty(total_length, dtype=torch.float32, device=self.device)

        # Load column 'V' of pd arrays in data to self.data
        index = 0
        for chunk in data:
            self.data[index : index + len(chunk)] = torch.from_numpy(
                chunk["V"].to_numpy()
            ).to(device=self.device)
            index += len(chunk)

        # Start the index of sampling at zero
        self.index = 0
        print("sMTJ Noise Data Loaded ...")

    def sample(self, sample_shape=[]):
        # Sample and deal with wrap around
        num_samples = math.prod(sample_shape)
        sample = self.data[self.index : self.index + num_samples]

        if num_samples > len(sample):
            sample = torch.cat((sample, self.data[: num_samples - len(sample)]), 0)

        self.index += num_samples
        if self.index > len(self.data):
            self.index = self.index - len(self.data)

        sample = self.scaling_factor * sample.view(sample_shape)
        sample = torch.sum(sample, axis=0)
        return sample  # Sum over first, n_sources, dimension


class simulatedMTJNoise(torch.distributions.distribution.Distribution):
    """Simulated MTJ Noise Distribution"""

    def __init__(
        self,
        scaling_factor=1.0,
        num_nodes=0,
        n_sources=1,
    ):
        print("Using Simulated MTJ Noise Distribution ...")
        assert (
            scaling_factor == 1.0
        ), "Scaling factor must be 1.0 if you wish to keep the scale of sMTJ noise equivalent to the dataset"
        self.scaling_factor = scaling_factor * 0.00090841466
        self.device = scaling_factor.device
        self.num_nodes = num_nodes
        self.n_sources = n_sources

        # The offsets are also computed based upon the sMTJ dataset
        self.offsets = (
            torch.sign(torch.randn(self.num_nodes, device=self.device)) * 0.0059316154
        )
        super(simulatedMTJNoise, self).__init__(validate_args=False)

    def __call__(self, sample_shape=[]):
        return self.sample([self.n_sources] + sample_shape)

    def sample(self, sample_shape=[]):
        assert sample_shape[-1] == self.num_nodes
        # Sample noise from Gaussian distribution
        noise = torch.randn(*sample_shape, device=self.device)

        offset_switch = (
            torch.rand([sample_shape[-2], sample_shape[-1]], device=self.device)
            < 0.0809
        )

        offset_switch = 2 * (offset_switch.float() - 0.5)
        offset_switch *= -1
        offset_switch = torch.cumprod(offset_switch, axis=0)
        offset_switch *= self.offsets[None, :]

        self.offsets = offset_switch[-1, :]

        # Apply scaling factor to noise and add offset
        sample = 0.042092104 + self.scaling_factor * noise + offset_switch[None, :, :]
        sample = torch.sum(sample, axis=0)
        return sample

if __name__ == "__main__":
    # Example usage

    distribution = simulatedMTJNoise(
        scaling_factor=torch.tensor([1.0]).to(torch.float32),
        num_nodes=10,
        n_sources=1,
    )
    samples = distribution(sample_shape=[1000, 10])

    # plot
    import matplotlib.pyplot as plt
    plt.hist(samples.cpu().numpy().flatten(), bins=50)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Simulated MTJ Noise Samples')
    plt.show()