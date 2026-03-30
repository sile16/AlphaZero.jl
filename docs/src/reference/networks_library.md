# [Networks Library](@id networks_library)

```@meta
CurrentModule = AlphaZero
```

For convenience, we provide a library of standard networks implementing the
[neural network interface](@ref network_interface).

These networks are contained in the `AlphaZero.NetLib` module, which resolves to
`AlphaZero.FluxLib` (the only available backend). The Knet backend was removed
due to incompatibility with recent Julia and CUDA versions.

## [Convolutional ResNet](@id conv_resnet)

```@docs
NetLib.ResNet
NetLib.ResNetHP
```

## [Simple Network](@id simplenet)

```@docs
NetLib.SimpleNet
NetLib.SimpleNetHP
```