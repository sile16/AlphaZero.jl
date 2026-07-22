"""Return whether every numeric leaf in a Flux gradient tree is finite.

Flux gradients are nested NamedTuples/Tuples whose numeric leaves are arrays.
`nothing` leaves represent non-trainable fields. Unknown metadata leaves are
ignored; numeric scalars and containers are always checked.
"""
_all_finite_gradient(::Nothing) = true
_all_finite_gradient(x::Number) = isfinite(x)
_all_finite_gradient(x::AbstractArray) = all(isfinite, x)
_all_finite_gradient(x::NamedTuple) = all(_all_finite_gradient, values(x))
_all_finite_gradient(x::Tuple) = all(_all_finite_gradient, x)
_all_finite_gradient(x::AbstractDict) = all(_all_finite_gradient, values(x))
_all_finite_gradient(_) = true
