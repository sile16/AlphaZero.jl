#####
##### Game trace
#####

"""
    Trace{State}

An object that collects all states visited during a game, along with the
rewards obtained at each step and the successive player policies to be used
as targets for the neural network.

For stochastic games, the `is_chance` field tracks which states were chance nodes.

# Constructor

    Trace(initial_state; is_chance_node=false)

"""
mutable struct Trace{State}
  states :: Vector{State}
  policies :: Vector{Vector{Float64}}
  rewards :: Vector{Float64}
  is_chance :: Vector{Bool}  # Track which states were chance nodes
  function Trace(init_state; is_chance_node=false)
    return new{typeof(init_state)}([init_state], [], [], [is_chance_node])
  end
end

function valid_trace(t::Trace)
  return length(t.policies) == length(t.rewards) == length(t.states) - 1 &&
         length(t.is_chance) == length(t.states)
end

"""
    Base.push!(t::Trace, π, r, s; is_chance=false)

Add a (target policy, reward, new state) quadruple to a trace.
For chance nodes, `is_chance` should be set to true (and π should be empty).
"""
function Base.push!(t::Trace, π, r, s; is_chance=false)
  push!(t.states, s)
  push!(t.policies, π)
  push!(t.rewards, r)
  push!(t.is_chance, is_chance)
end

function Base.length(t::Trace)
  return length(t.rewards)
end

function total_reward(t::Trace, gamma=1.)
  return sum([gamma^(i-1) * r for (i, r) in enumerate(t.rewards)])
end

function debug_trace(gspec::AbstractGameSpec, t::Trace)
  n = length(t)
  for i in 1:n
    println("Transition $i:")
    game = GI.init(gspec, t.states[i])
    GI.render(game)
    for (a, p) in zip(GI.available_actions(game),  t.policies[i])
      print("$(GI.action_string(gspec, a)): $(pyfmt(".3f", p))  ")
    end
    println("")
    println("Obtained reward of: $(t.rewards[i]).")
    println("")
  end
  println("Showing final state:")
  GI.render(GI.init(gspec, t.states[n + 1]))
end
