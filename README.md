# Code accompanying the paper [Noise-based Local Learning using Stochastic Magnetic Tunnel Junctions](https://arxiv.org/abs/2412.12783)
## Configuration
Configuration is done with Hydra.
When running real-world noise experiments, the noise file needs to be provided in the config.
For access to the sMTJ data, please get in touch with the author at kees.koenders@donders.ru.nl.

All configs used to produce the results in the paper can be found in conf/runs/.

## Required Packages
PyTorch (torch, torchvision)\
hydra-core\
numpy\
pandas\
wandb\
matplotlib\
