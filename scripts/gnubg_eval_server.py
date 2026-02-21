#!/usr/bin/env python3
"""GnuBG evaluation server — reads board positions from stdin, returns probabilities.

Protocol (one JSON per line):
  Request:  {"boards": [[[25 ints], [25 ints]], ...], "ply": 0}
  Response: {"probs": [[win, wg, wbg, lg, lbg], ...]}

  Quit:     {"cmd": "quit"}
  Response: (none, process exits)
"""

import sys
import json
import gnubg

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        print(json.dumps({"error": "invalid json"}), flush=True)
        continue

    if "cmd" in data:
        if data["cmd"] == "quit":
            break
        continue

    boards = data["boards"]
    ply = data.get("ply", 0)

    results = []
    for board in boards:
        probs = gnubg.probabilities(board, ply)
        results.append(list(probs))

    print(json.dumps({"probs": results}), flush=True)
