# Package Overview

The philosophy of this project is to provide an implementation of AlphaZero that
is simple enough to be widely accessible for students and researchers, while
also being sufficiently powerful and fast to enable meaningful experiments on
limited computing resources.

On this page, we describe some key features of AlphaZero.jl:
  - A standalone and straightforward [MCTS implementation](@ref mcts)
  - Some facilities for asynchronous and distributed simulation
  - A series of opt-in optimizations to increase training efficiency
  - Generic interfaces for games and neural networks
  - A simple user interface to get started quickly and diagnose problems

### Batched MCTS and Distributed Training

A key aspect of generating self-play data efficiently is to simulate a large number
of games and batch neural network evaluations across MCTS simulations. AlphaZero.jl
provides a `BatchedMCTS` module that runs multiple MCTS simulations in lockstep,
collecting leaf positions into a single batch for efficient CPU or GPU inference.

Training uses an HTTP-based distributed architecture: a training server (with GPU)
manages the replay buffer and runs gradient updates, while self-play clients
generate games using CPU inference and upload samples. This scales from a single
machine (server + client on localhost) to multiple machines over a network.

### Training Optimizations

AlphaZero.jl has out-of-the-box support for many of the optimizations introduced
in [Oracle's
series](https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191)
and also implements new ones. These include:

- Position averaging
- Memory buffer with growing window
- Cyclical learning rates
- Symmetry-based data augmentation and game randomization

All these optimizations are documented in the [Training Parameters](@ref params)
section of the manual.

### Game Interface

You can use AlphaZero.jl on the game of your choice by simply implementing the
[Game Interface](@ref game_interface). Currently, there is support for
two-players, zero-sum games with finite action spaces and perfect information.
Support for Markov Decision Processes will be added in a forthcoming release.

Please see [here](@ref own_game) for recommendations on how to use AlphaZero.jl on
your own game.

### Network Interface

AlphaZero.jl allows you to plug any neural network that implements the
[Network Interface](@ref network_interface). We provide a [library](@ref
networks_library) of standard Flux-based networks, including two-headed
[multi-layer perceptrons](@ref simplenet) and
[convolutional resnets](@ref conv_resnet).

### User Interface and Utilities

AlphaZero.jl comes with _batteries included_. It features a simple [user
interface](@ref ui) along with utilities for session management, logging,
profiling, benchmarking and model exploration.

- A [session management](@ref session) system makes it easy to interrupt and
  resume training.
- An [interactive command interpreter](@ref explorer) can be used to explore the
  behavior of AlphaZero agents.
- Common tasks can be executed in a single line thanks to the [Scripts](@ref scripts)
  module.
- Reports are generated automatically after each training iteration to help
  diagnosing problems and tuning hyperparameters. An extensive documentation of
  collected metrics can be found in [Training Reports](@ref reports) and
  [Benchmarks](@ref benchmark).

Finally, because the user interface is implemented separately from the core
algorithm, it can be extended or [replaced](@ref improve_ui) easily.

```@raw html
<div>
<img src="../../assets/img/connect-four/plots/benchmark_won_games.png" width="24%" />
<img src="../../assets/img/connect-four/plots/arena.png" width="24%" />
<img src="../../assets/img/connect-four/plots/exploration_depth.png" width="24%" />
<img src="../../assets/img/connect-four/plots/entropies.png" width="24%" />
<img src="../../assets/img/connect-four/plots/loss.png" width="24%" />
<img src="../../assets/img/connect-four/plots/loss_per_stage.png" width="24%"/>
<img src="../../assets/img/connect-four/plots/iter_perfs/1.png" width="24%"/>
<img src="../../assets/img/connect-four/plots/iter_loss/1.png" width="24%" />
<img src="../../assets/img/ui-first-iter-cut.png" width="48%" />
<img src="../../assets/img/explorer.png" width="48%" />
<!--<img src="../../assets/img/connect-four/plots/iter_summary/1.png" width="24%" />-->
</div>
```
