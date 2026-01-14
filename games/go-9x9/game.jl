"""
9x9 Go implementation for AlphaZero.jl

Rules:
- Chinese scoring (area scoring)
- Simple ko rule
- Komi: 7.5 points for White
- Superko not implemented (simple ko only)
- Game ends after two consecutive passes
"""

import AlphaZero.GI
using StaticArrays
using Crayons

const BOARD_SIZE = 9
const NUM_INTERSECTIONS = BOARD_SIZE * BOARD_SIZE  # 81
const KOMI = 7.5f0  # Compensation for White
const MAX_MOVES = 200  # Prevent infinite games (typical 9x9 games are 50-100 moves)

# Board cell states
const EMPTY = UInt8(0)
const BLACK = UInt8(1)
const WHITE = UInt8(2)

# Action encoding: 0-80 for board positions, 81 for pass
const PASS_ACTION = NUM_INTERSECTIONS + 1  # 82
const NUM_ACTIONS = NUM_INTERSECTIONS + 1  # 82 total actions

const ACTIONS = collect(1:NUM_ACTIONS)

# Direction offsets for finding neighbors
const DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

#####
##### Board representation
#####

struct Board
  stones :: SMatrix{BOARD_SIZE, BOARD_SIZE, UInt8, NUM_INTERSECTIONS}
end

const EMPTY_BOARD = Board(@SMatrix zeros(UInt8, BOARD_SIZE, BOARD_SIZE))

#####
##### Game state
#####

struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv
  board :: Board
  curplayer :: UInt8  # BLACK or WHITE
  ko_point :: Int  # 0 if no ko, otherwise the forbidden point (1-81)
  passes :: Int  # Number of consecutive passes
  black_captures :: Int  # Stones captured by Black
  white_captures :: Int  # Stones captured by White
  move_count :: Int  # Total moves played (to enforce MAX_MOVES)
  finished :: Bool
end

function GI.init(::GameSpec)
  GameEnv(EMPTY_BOARD, BLACK, 0, 0, 0, 0, 0, false)
end

GI.spec(::GameEnv) = GameSpec()
GI.two_players(::GameSpec) = true
GI.actions(::GameSpec) = ACTIONS

#####
##### Coordinate utilities
#####

# Convert 1D action (1-81) to 2D coordinates (row, col) - both 1-indexed
function action_to_coords(action::Int)
  @assert 1 <= action <= NUM_INTERSECTIONS
  row = (action - 1) ÷ BOARD_SIZE + 1
  col = (action - 1) % BOARD_SIZE + 1
  return (row, col)
end

# Convert 2D coordinates to 1D action
function coords_to_action(row::Int, col::Int)
  return (row - 1) * BOARD_SIZE + col
end

# Check if coordinates are on the board
function on_board(row::Int, col::Int)
  return 1 <= row <= BOARD_SIZE && 1 <= col <= BOARD_SIZE
end

# Get stone at position
function get_stone(board::Board, row::Int, col::Int)
  return board.stones[row, col]
end

function get_stone(board::Board, action::Int)
  row, col = action_to_coords(action)
  return get_stone(board, row, col)
end

# Set stone at position (returns new board)
function set_stone(board::Board, row::Int, col::Int, stone::UInt8)
  new_stones = setindex(board.stones, stone, row, col)
  return Board(new_stones)
end

#####
##### Liberty counting and capture detection
#####

"""
Find all stones in the group containing (row, col) and count liberties.
Returns (group_positions, liberty_count)
"""
function find_group_and_liberties(board::Board, row::Int, col::Int)
  stone = get_stone(board, row, col)
  if stone == EMPTY
    return Int[], 0
  end

  group = Int[]
  liberties = Set{Int}()
  visited = falses(BOARD_SIZE, BOARD_SIZE)

  # BFS/DFS to find connected group
  stack = [(row, col)]

  while !isempty(stack)
    r, c = pop!(stack)

    if visited[r, c]
      continue
    end
    visited[r, c] = true

    push!(group, coords_to_action(r, c))

    # Check all neighbors
    for (dr, dc) in DIRECTIONS
      nr, nc = r + dr, c + dc
      if on_board(nr, nc)
        neighbor = get_stone(board, nr, nc)
        if neighbor == EMPTY
          push!(liberties, coords_to_action(nr, nc))
        elseif neighbor == stone && !visited[nr, nc]
          push!(stack, (nr, nc))
        end
      end
    end
  end

  return group, length(liberties)
end

"""
Remove a group of stones from the board.
Returns (new_board, num_captured)
"""
function remove_group(board::Board, group::Vector{Int})
  new_board = board
  for action in group
    row, col = action_to_coords(action)
    new_board = set_stone(new_board, row, col, EMPTY)
  end
  return new_board, length(group)
end

"""
Check if placing a stone at (row, col) would capture any opponent groups.
Returns list of groups (each group is a list of positions) that would be captured.
"""
function find_captures(board::Board, row::Int, col::Int, player::UInt8)
  opponent = player == BLACK ? WHITE : BLACK
  captures = Vector{Int}[]
  checked = Set{Int}()

  for (dr, dc) in DIRECTIONS
    nr, nc = row + dr, col + dc
    if on_board(nr, nc)
      action = coords_to_action(nr, nc)
      if get_stone(board, nr, nc) == opponent && !(action in checked)
        group, liberties = find_group_and_liberties(board, nr, nc)
        for pos in group
          push!(checked, pos)
        end
        # After placing our stone, this neighbor loses one liberty
        # Check if it would have zero liberties
        if liberties == 1  # The only liberty is where we're placing
          push!(captures, group)
        end
      end
    end
  end

  return captures
end

"""
Check if a move is suicide (placing stone that immediately dies without capturing).
"""
function is_suicide(board::Board, row::Int, col::Int, player::UInt8)
  # Temporarily place the stone
  temp_board = set_stone(board, row, col, player)

  # Check if this move captures anything
  captures = find_captures(board, row, col, player)
  if !isempty(captures)
    return false  # Not suicide if it captures
  end

  # Check if the placed stone's group has liberties
  _, liberties = find_group_and_liberties(temp_board, row, col)
  return liberties == 0
end

#####
##### Move execution
#####

function GI.play!(g::GameEnv, action::Int)
  @assert !g.finished

  g.move_count += 1

  if action == PASS_ACTION
    g.passes += 1
    if g.passes >= 2 || g.move_count >= MAX_MOVES
      g.finished = true
    end
    g.ko_point = 0
    g.curplayer = g.curplayer == BLACK ? WHITE : BLACK
    return
  end

  # Reset pass counter
  g.passes = 0

  row, col = action_to_coords(action)
  player = g.curplayer
  opponent = player == BLACK ? WHITE : BLACK

  # Place the stone
  g.board = set_stone(g.board, row, col, player)

  # Find and remove captured stones
  total_captured = 0
  potential_ko_point = 0

  for (dr, dc) in DIRECTIONS
    nr, nc = row + dr, col + dc
    if on_board(nr, nc) && get_stone(g.board, nr, nc) == opponent
      group, liberties = find_group_and_liberties(g.board, nr, nc)
      if liberties == 0
        g.board, num_captured = remove_group(g.board, group)
        total_captured += num_captured
        # Track for ko detection
        if num_captured == 1
          potential_ko_point = group[1]
        end
      end
    end
  end

  # Update capture counts
  if player == BLACK
    g.black_captures += total_captured
  else
    g.white_captures += total_captured
  end

  # Ko detection: if we captured exactly one stone, the position where it was
  # might be ko (opponent can't immediately recapture)
  if total_captured == 1
    # Check if our placed stone has exactly one liberty (the captured position)
    _, our_liberties = find_group_and_liberties(g.board, row, col)
    if our_liberties == 1
      g.ko_point = potential_ko_point
    else
      g.ko_point = 0
    end
  else
    g.ko_point = 0
  end

  g.curplayer = opponent

  # Check move limit
  if g.move_count >= MAX_MOVES
    g.finished = true
  end
end

#####
##### Legal moves
#####

function GI.actions_mask(g::GameEnv)
  mask = falses(NUM_ACTIONS)

  if g.finished
    return mask
  end

  player = g.curplayer

  for action in 1:NUM_INTERSECTIONS
    row, col = action_to_coords(action)

    # Must be empty
    if get_stone(g.board, row, col) != EMPTY
      continue
    end

    # Can't play on ko point
    if action == g.ko_point
      continue
    end

    # Can't be suicide
    if is_suicide(g.board, row, col, player)
      continue
    end

    mask[action] = true
  end

  # Pass is always legal
  mask[PASS_ACTION] = true

  return mask
end

#####
##### Game state
#####

function GI.current_state(g::GameEnv)
  return (
    board = g.board,
    curplayer = g.curplayer,
    ko_point = g.ko_point,
    passes = g.passes,
    move_count = g.move_count
  )
end

function GI.set_state!(g::GameEnv, state)
  g.board = state.board
  g.curplayer = state.curplayer
  g.ko_point = state.ko_point
  g.passes = state.passes
  g.move_count = state.move_count
  g.finished = state.passes >= 2 || state.move_count >= MAX_MOVES
end

GI.white_playing(g::GameEnv) = g.curplayer == WHITE

GI.game_terminated(g::GameEnv) = g.finished

#####
##### Scoring (Chinese/Area scoring)
#####

"""
Count territory using flood fill from empty points.
Returns (black_territory, white_territory)
"""
function count_territory(board::Board)
  visited = falses(BOARD_SIZE, BOARD_SIZE)
  black_territory = 0
  white_territory = 0

  for row in 1:BOARD_SIZE
    for col in 1:BOARD_SIZE
      if visited[row, col] || get_stone(board, row, col) != EMPTY
        continue
      end

      # Flood fill to find connected empty region
      region = Int[]
      borders_black = false
      borders_white = false
      stack = [(row, col)]

      while !isempty(stack)
        r, c = pop!(stack)

        if visited[r, c]
          continue
        end

        stone = get_stone(board, r, c)

        if stone == BLACK
          borders_black = true
          continue
        elseif stone == WHITE
          borders_white = true
          continue
        end

        visited[r, c] = true
        push!(region, coords_to_action(r, c))

        for (dr, dc) in DIRECTIONS
          nr, nc = r + dr, c + dc
          if on_board(nr, nc) && !visited[nr, nc]
            push!(stack, (nr, nc))
          end
        end
      end

      # Assign territory
      if borders_black && !borders_white
        black_territory += length(region)
      elseif borders_white && !borders_black
        white_territory += length(region)
      end
      # If borders both or neither, it's neutral (dame)
    end
  end

  return black_territory, white_territory
end

"""
Chinese scoring: stones on board + surrounded territory
"""
function compute_score(g::GameEnv)
  # Count stones on board
  black_stones = 0
  white_stones = 0

  for row in 1:BOARD_SIZE
    for col in 1:BOARD_SIZE
      stone = get_stone(g.board, row, col)
      if stone == BLACK
        black_stones += 1
      elseif stone == WHITE
        white_stones += 1
      end
    end
  end

  # Count territory
  black_territory, white_territory = count_territory(g.board)

  black_score = Float32(black_stones + black_territory)
  white_score = Float32(white_stones + white_territory) + KOMI

  return black_score, white_score
end

function GI.white_reward(g::GameEnv)
  if !g.finished
    return 0.0f0
  end

  black_score, white_score = compute_score(g)

  if white_score > black_score
    return 1.0f0
  elseif black_score > white_score
    return -1.0f0
  else
    return 0.0f0  # Draw (very rare with 0.5 komi)
  end
end

#####
##### Heuristic for MinMax
#####

function GI.heuristic_value(g::GameEnv)
  black_score, white_score = compute_score(g)
  # Normalize to roughly [-1, 1]
  diff = (white_score - black_score) / 40.0f0
  return g.curplayer == WHITE ? diff : -diff
end

#####
##### Neural network interface
#####

function GI.vectorize_state(::GameSpec, state)
  board = state.board
  curplayer = state.curplayer

  # 4 channels: current player stones, opponent stones, empty, ko point
  # Alternatively: black stones, white stones, current player indicator, ko
  # Let's use: current player stones, opponent stones, ones (bias), ko point

  current_stones = zeros(Float32, BOARD_SIZE, BOARD_SIZE)
  opponent_stones = zeros(Float32, BOARD_SIZE, BOARD_SIZE)
  ko_plane = zeros(Float32, BOARD_SIZE, BOARD_SIZE)

  current_color = curplayer
  opponent_color = curplayer == BLACK ? WHITE : BLACK

  for row in 1:BOARD_SIZE
    for col in 1:BOARD_SIZE
      stone = get_stone(board, row, col)
      if stone == current_color
        current_stones[row, col] = 1.0f0
      elseif stone == opponent_color
        opponent_stones[row, col] = 1.0f0
      end
    end
  end

  # Mark ko point
  if state.ko_point > 0
    row, col = action_to_coords(state.ko_point)
    ko_plane[row, col] = 1.0f0
  end

  # Stack into 3D array: (BOARD_SIZE, BOARD_SIZE, 3)
  return cat(current_stones, opponent_stones, ko_plane; dims=3)
end

#####
##### Symmetries (8-fold: rotations and reflections)
#####

function rotate_board_90(stones::SMatrix)
  # Rotate 90 degrees clockwise
  new_stones = similar(stones)
  for row in 1:BOARD_SIZE
    for col in 1:BOARD_SIZE
      new_stones[col, BOARD_SIZE - row + 1] = stones[row, col]
    end
  end
  return SMatrix{BOARD_SIZE, BOARD_SIZE, UInt8, NUM_INTERSECTIONS}(new_stones)
end

function flip_board_horizontal(stones::SMatrix)
  new_stones = similar(stones)
  for row in 1:BOARD_SIZE
    for col in 1:BOARD_SIZE
      new_stones[row, BOARD_SIZE - col + 1] = stones[row, col]
    end
  end
  return SMatrix{BOARD_SIZE, BOARD_SIZE, UInt8, NUM_INTERSECTIONS}(new_stones)
end

# Transform action index according to board transformation
function transform_action(action::Int, transform_fn)
  if action == PASS_ACTION
    return PASS_ACTION
  end
  row, col = action_to_coords(action)
  # Apply same transform logic to coordinates
  # This is complex - for now, skip symmetries in training
  return action
end

#####
##### User interface
#####

const COL_LABELS = "ABCDEFGHJ"  # Skip 'I' as per Go convention

function GI.action_string(::GameSpec, action::Int)
  if action == PASS_ACTION
    return "pass"
  end
  row, col = action_to_coords(action)
  return "$(COL_LABELS[col])$(row)"
end

function GI.parse_action(::GameSpec, str::String)
  str = strip(lowercase(str))

  if str == "pass" || str == "p"
    return PASS_ACTION
  end

  length(str) >= 2 || return nothing

  col_char = uppercase(str[1])
  col_idx = findfirst(==(col_char), COL_LABELS)
  isnothing(col_idx) && return nothing

  row = tryparse(Int, str[2:end])
  isnothing(row) && return nothing

  1 <= row <= BOARD_SIZE || return nothing

  return coords_to_action(row, col_idx)
end

function GI.render(g::GameEnv; botmargin=true)
  println()

  # Column headers
  print("   ")
  for col in 1:BOARD_SIZE
    print(" $(COL_LABELS[col])")
  end
  println()

  # Board rows (from top to bottom, so row 9 first)
  for row in BOARD_SIZE:-1:1
    print(lpad(row, 2), " ")
    for col in 1:BOARD_SIZE
      stone = get_stone(g.board, row, col)
      action = coords_to_action(row, col)

      if stone == BLACK
        print(crayon"black bold", " X", crayon"reset")
      elseif stone == WHITE
        print(crayon"white bold", " O", crayon"reset")
      elseif action == g.ko_point
        print(crayon"red", " *", crayon"reset")  # Ko point
      else
        # Show star points (hoshi)
        is_star = (row, col) in [(3,3), (3,7), (5,5), (7,3), (7,7)]
        if is_star
          print(crayon"yellow", " +", crayon"reset")
        else
          print(" .")
        end
      end
    end
    println(" ", row)
  end

  # Column footers
  print("   ")
  for col in 1:BOARD_SIZE
    print(" $(COL_LABELS[col])")
  end
  println()

  # Game info
  player = g.curplayer == BLACK ? "Black (X)" : "White (O)"
  println("\nTo play: $player")
  println("Captures - Black: $(g.black_captures), White: $(g.white_captures)")

  if g.passes > 0
    println("Consecutive passes: $(g.passes)")
  end

  if g.finished
    black_score, white_score = compute_score(g)
    println("\nGame Over!")
    println("Black: $black_score, White: $white_score (incl. $(KOMI) komi)")
    if white_score > black_score
      println("White wins by $(white_score - black_score) points")
    elseif black_score > white_score
      println("Black wins by $(black_score - white_score) points")
    else
      println("Draw")
    end
  end

  botmargin && println()
end

function GI.read_state(::GameSpec)
  # Not implementing full state reading for Go
  return nothing
end
