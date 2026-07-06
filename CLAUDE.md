# AlphaZero Agent Notes

Use `AGENTS.md` as the durable integration contract for BackgammonNet,
distributed training, bearoff tables, and cube-scope decisions.

## Common Commands

```bash
julia --project -e 'using Pkg; Pkg.test()'
julia --project test/runtests.jl
julia --threads auto --project scripts/training_server.jl --data-dir ./sessions/alphazero-server
julia --threads auto --project scripts/selfplay_client.jl --server http://jarvis:9090
```

## Path Policy

Active code should use repo-relative paths or explicit environment variables.
Backgammon bearoff artifact discovery must go through BackgammonNet helper APIs,
not machine-specific absolute paths.
