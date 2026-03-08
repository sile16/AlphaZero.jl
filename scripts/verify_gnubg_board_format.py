#!/usr/bin/env python3
"""
Verify gnubg's board format by examining known positions.

Key questions:
1. What does index 0 represent? (ace point or bar?)
2. What does index 24 represent? (bar or 24-point?)
3. Do both arrays use the same perspective or each player's own perspective?
"""

import gnubg

print("=" * 70)
print("GNUBG BOARD FORMAT VERIFICATION")
print("=" * 70)

position_id = '4HPwATDgc/ABMA'

print(f"\nPosition ID: {position_id}")
print("(Standard backgammon opening position)")

board = gnubg.board_from_position_id(position_id)

print(f"\nboard type: {type(board)}")
print(f"board length: {len(board)}")
print(f"board[0] length: {len(board[0])}")
print(f"board[1] length: {len(board[1])}")

print("\n" + "=" * 70)
print("FULL BOARD ARRAYS")
print("=" * 70)

print("\nboard[0] (first player array):")
print(f"  Raw: {list(board[0])}")
for i in range(len(board[0])):
    val = board[0][i]
    if val > 0:
        label = f"  index {i:2d}: {val} checkers"
        if i == 0:
            label += "  <-- INDEX 0"
        elif i == 24:
            label += "  <-- INDEX 24"
        print(label)

print(f"\nboard[1] (second player array):")
print(f"  Raw: {list(board[1])}")
for i in range(len(board[1])):
    val = board[1][i]
    if val > 0:
        label = f"  index {i:2d}: {val} checkers"
        if i == 0:
            label += "  <-- INDEX 0"
        elif i == 24:
            label += "  <-- INDEX 24"
        print(label)

total_0 = sum(board[0])
total_1 = sum(board[1])
print(f"\nTotal checkers in board[0]: {total_0}")
print(f"Total checkers in board[1]: {total_1}")

print("\n" + "=" * 70)
print("ANALYSIS: Testing interpretations")
print("=" * 70)

print("""
Standard opening (from absolute point numbering, P0 perspective):
  Player 0 has: 2@pt24, 5@pt13, 3@pt8, 5@pt6    (moves 24->1->off)
  Player 1 has: 2@pt1, 5@pt12, 3@pt17, 5@pt19    (moves 1->24->off)
""")

# ---- Interpretation A: idx0=ace point (0-indexed), idx24=bar, OWN perspective ----
# Each player's array in their OWN perspective (mirrored for P1)
# "own perspective" means: player's home board = low indices
# P0: pt6->idx5, pt8->idx7, pt13->idx12, pt24->idx23
# P1: P0's pt1->P1's pt24->idx23, P0's pt12->P1's pt13->idx12,
#     P0's pt17->P1's pt8->idx7, P0's pt19->P1's pt6->idx5
expA_0 = [0]*25
expA_0[5] = 5; expA_0[7] = 3; expA_0[12] = 5; expA_0[23] = 2
expA_1 = [0]*25
expA_1[5] = 5; expA_1[7] = 3; expA_1[12] = 5; expA_1[23] = 2

print("Interp A: idx0=ace (0-indexed), idx24=bar, each player OWN perspective")
print(f"  Expected board[0]: {expA_0}")
print(f"  Actual   board[0]: {list(board[0])}")
print(f"  Match: {list(board[0]) == expA_0}")
print(f"  Expected board[1]: {expA_1}")
print(f"  Actual   board[1]: {list(board[1])}")
print(f"  Match: {list(board[1]) == expA_1}")

# ---- Interpretation B: idx0=bar, idx1-24=points (1-indexed), OWN perspective ----
expB_0 = [0]*25
expB_0[6] = 5; expB_0[8] = 3; expB_0[13] = 5; expB_0[24] = 2
expB_1 = [0]*25
expB_1[6] = 5; expB_1[8] = 3; expB_1[13] = 5; expB_1[24] = 2

print(f"\nInterp B: idx0=bar, idx1-24=points 1-24 (1-indexed), each player OWN perspective")
print(f"  Expected board[0]: {expB_0}")
print(f"  Actual   board[0]: {list(board[0])}")
print(f"  Match: {list(board[0]) == expB_0}")
print(f"  Expected board[1]: {expB_1}")
print(f"  Actual   board[1]: {list(board[1])}")
print(f"  Match: {list(board[1]) == expB_1}")

# ---- Interpretation C: idx0=ace (0-indexed), idx24=bar, P0's absolute perspective ----
# board[0] = P0's checkers in P0's point numbering
# board[1] = P1's checkers in P0's point numbering
expC_0 = [0]*25
expC_0[5] = 5; expC_0[7] = 3; expC_0[12] = 5; expC_0[23] = 2
expC_1 = [0]*25
expC_1[0] = 2; expC_1[11] = 5; expC_1[16] = 3; expC_1[18] = 5

print(f"\nInterp C: idx0=ace (0-indexed), idx24=bar, BOTH in P0's absolute perspective")
print(f"  Expected board[0]: {expC_0}")
print(f"  Actual   board[0]: {list(board[0])}")
print(f"  Match: {list(board[0]) == expC_0}")
print(f"  Expected board[1]: {expC_1}")
print(f"  Actual   board[1]: {list(board[1])}")
print(f"  Match: {list(board[1]) == expC_1}")

# ---- Interpretation D: idx0=bar, idx1-24=points (1-indexed), P0's absolute perspective ----
expD_0 = [0]*25
expD_0[6] = 5; expD_0[8] = 3; expD_0[13] = 5; expD_0[24] = 2
expD_1 = [0]*25
expD_1[1] = 2; expD_1[12] = 5; expD_1[17] = 3; expD_1[19] = 5

print(f"\nInterp D: idx0=bar, idx1-24=points 1-24 (1-indexed), BOTH in P0's absolute perspective")
print(f"  Expected board[0]: {expD_0}")
print(f"  Actual   board[0]: {list(board[0])}")
print(f"  Match: {list(board[0]) == expD_0}")
print(f"  Expected board[1]: {expD_1}")
print(f"  Actual   board[1]: {list(board[1])}")
print(f"  Match: {list(board[1]) == expD_1}")

print("\n" + "=" * 70)
print("ADDITIONAL TESTS")
print("=" * 70)

# Try gnubg.board() if available
try:
    b = gnubg.board()
    print(f"\ngnubg.board() returned: {b}")
except Exception as e:
    print(f"\ngnubg.board() raised: {type(e).__name__}: {e}")

# Try to see gnubg help/dir
print(f"\ngnubg module attributes: {[x for x in dir(gnubg) if not x.startswith('_')]}")

# Try position_id_from_board to verify round-trip
try:
    rt_id = gnubg.position_id_from_board(board)
    print(f"\nRound-trip: position_id_from_board(board) = '{rt_id}'")
    print(f"Original position_id = '{position_id}'")
    print(f"Match: {rt_id == position_id}")
except Exception as e:
    print(f"\nposition_id_from_board raised: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nboard[0] non-zero: {[(i, board[0][i]) for i in range(len(board[0])) if board[0][i] > 0]}")
print(f"board[1] non-zero: {[(i, board[1][i]) for i in range(len(board[1])) if board[1][i] > 0]}")

# Determine which interpretation is correct
actual_0 = list(board[0])
actual_1 = list(board[1])

if actual_0 == expA_0 and actual_1 == expA_1:
    print("\n>>> CONFIRMED: Interpretation A (idx0=ace 0-indexed, idx24=bar, OWN perspective)")
    print("    Both arrays are SYMMETRIC (same pattern) because opening position is symmetric!")
    print("    Index 0 = ace point (point 1 in own perspective)")
    print("    Index 24 = bar")
    print("    Each array uses THAT PLAYER's own perspective")
elif actual_0 == expB_0 and actual_1 == expB_1:
    print("\n>>> CONFIRMED: Interpretation B (idx0=bar, idx1-24=points, OWN perspective)")
elif actual_0 == expC_0 and actual_1 == expC_1:
    print("\n>>> CONFIRMED: Interpretation C (idx0=ace 0-indexed, idx24=bar, P0's absolute perspective)")
elif actual_0 == expD_0 and actual_1 == expD_1:
    print("\n>>> CONFIRMED: Interpretation D (idx0=bar, idx1-24=points, P0's absolute perspective)")
else:
    print("\n>>> NO INTERPRETATION MATCHED! Need to investigate further.")
    
    # Note: opening position is symmetric, so OWN perspective makes both arrays identical
    # Let's check if they're the same
    if actual_0 == actual_1:
        print("    NOTE: board[0] == board[1] (arrays are identical)")
        print("    This is expected for opening position with OWN perspective (symmetric)")
        print("    We need an asymmetric position to distinguish A from B")
