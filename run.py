import nodeperturbation.utils as utils
import nodeperturbation
from nodeperturbation.np import NPLinear
from nodeperturbation.bp import BPLinear
import torchvision
import torch
import numpy as np
from tqdm import tqdm
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base="1.3", config_path="conf/", config_name="default_config")
def run(config: DictConfig) -> None:
    cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    wandb.init(
        config=cfg,
        entity=config.wandb.entity,
        project=config.wandb.project,
        mode=config.wandb.mode,
    )

    assert config.dataset in [
        "MNIST",
        "CIFAR10",
        "CIFAR100",
        "TIN",
    ], "Dataset must be MNIST, CIFAR10, CIFAR100, or TIN (tinyimagenet)"

    assert config.optimizer.loss_func in ["CCE", "MSE"], "Dataset must be MSE or CCE"

    assert config.noise.distribution in [
        "sMTJ",
        "Normal",
        "Bernoulli",
        "CentredBernoulli",
        "simulatedMTJ",
    ], "Noise distribution must be Normal, Bernoulli, CentredBernoulli, sMTJ, or simulatedMTJ"

    assert config.learning_rule in [
        "NP",
        "BP",
    ], "Layer type must be NP or BP"

    assert config.optimizer.type in [
        "Adam",
        "SGD",
    ], "Optimizer must be Adam, or SGD"

    if config.learning_rule == "BP":
        assert config.noise.sigma == 0.0, "Sigma cannot be non-zero for BP."

    if config.optimizer.betas is None or config.optimizer.eps is None:
        assert config.optimizer.type != "Adam", "Epsilon/Betas must be defined for Adam"

    # Initializing random seeding
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load dataset
    tv_dataset = config.dataset
    if config.dataset == "MNIST":
        tv_dataset = torchvision.datasets.MNIST
    elif config.dataset == "CIFAR10":
        tv_dataset = torchvision.datasets.CIFAR10
    elif config.dataset == "CIFAR100":
        tv_dataset = torchvision.datasets.CIFAR100

    layer_mapping = {
        "NP": NPLinear,
        "BP": BPLinear,
    }
    layer_type = layer_mapping[config.learning_rule]

    train_loader, test_loader = utils.construct_dataloaders(
        tv_dataset, batch_size=config.training.batch_size, device=config.device
    )

    # If dataset is CIFAR, change input shape
    in_size = 28 * 28
    out_size = 10
    if tv_dataset == torchvision.datasets.CIFAR10:
        in_size = 32 * 32 * 3
    if tv_dataset == torchvision.datasets.CIFAR100:
        in_size = 32 * 32 * 3
        out_size = 100
    if tv_dataset == "TIN":
        in_size = 64 * 64 * 3
        out_size = 200

    # Initialize model
    model = nodeperturbation.net.construct_net(
        in_size=in_size,
        out_size=out_size,
        layer_type=layer_type,
        config=config,
    )
    model.to(config.device)

    # Initialize metric storage
    metrics = utils.init_metric()

    # Define optimizers
    fwd_optimizer = None
    if config.optimizer.type == "Adam":
        fwd_optimizer = torch.optim.Adam(
            model.get_fwd_params(),
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            lr=config.optimizer.fwd_lr,
        )
    elif config.optimizer.type == "SMORMS3":
        fwd_optimizer = utils.SMORMS3(
            model.get_fwd_params(), lr=config.optimizer.fwd_lr
        )
    elif config.optimizer.type == "SGD":
        fwd_optimizer = torch.optim.SGD(
            model.get_fwd_params(), lr=config.optimizer.fwd_lr
        )

    optimizers = [fwd_optimizer]
    if config.optimizer.decor_lr != 0.0:
        decor_optimizer = torch.optim.SGD(
            model.get_decor_params(), lr=config.optimizer.decor_lr
        )
        optimizers.append(decor_optimizer)

    loss_func = None
    if config.optimizer.loss_func == "CCE":
        loss_obj = torch.nn.CrossEntropyLoss(reduction="none")
        loss_func = lambda input, target, onehot: loss_obj(input, target)
    elif config.optimizer.loss_func == "MSE":
        loss_obj = torch.nn.MSELoss(reduction="none")
        loss_func = lambda input, target, onehot: torch.sum(
            loss_obj(input, onehot), axis=1
        )

    # Train loop
    for e in tqdm(range(config.training.nb_epochs + 1)):
        metrics = utils.update_metrics(
            model,
            metrics,
            config.device,
            "train",
            train_loader,
            loss_func,
            e,
            loud=True,
            wandb=wandb,
            num_classes=out_size,
        )
        metrics = utils.update_metrics(
            model,
            metrics,
            config.device,
            "test",
            test_loader,
            loss_func,
            e,
            loud=False,
            wandb=wandb,
            num_classes=out_size,
        )
        if e < config.training.nb_epochs:
            utils.train(
                model,
                config.device,
                train_loader,
                optimizers,
                e,
                loss_func,
                loud=False,
                num_classes=out_size,
            )
        if np.isnan(metrics["test"]["loss"][-1]) or np.isnan(
            metrics["train"]["loss"][-1]
        ):
            print("NaN detected, aborting training")
            break

    wandb.finish()


if __name__ == "__main__":
    run()
