# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- Run anything that compiles the full 0.25° model (≈1 min compile, then GPU-heavy) as a
  background task with a completion notification, and keep foreground commands short so the
  user's messages get through.
- The development machine's GPU may be shared with other workloads: check `nvidia-smi` and ask
  before starting long GPU runs.
- Cloud runs (Modal, vast.ai): state the estimated cost first, stop apps explicitly afterwards
  (`modal app stop -y <id>`; detached apps can linger in "ephemeral"), destroy rented
  instances when done, and never print or commit API keys (`~/.modal.toml`,
  `~/.config/vastai/vast_api_key`).
- When a result will be quoted in docs, save the raw output (JSON/log) next to the script that
  produced it and link it from docs/validation.md.
