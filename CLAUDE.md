# Project description
See @README.md

# Running the program
See @README.md Usage section for standard run commands. To do a fresh run (wipe DB but keep reports):
```bash
rm -f "$DB_PATH" && uv run python analyze.py -p portfolio.txt --verbose
```

# Rules
* Always run Python via `uv run python` — never bare `python` or `python3`.
* After completing any implementation task, check the README for discrepancies (architecture diagram, features list, project structure) and update it before considering the task done.
* Commit after every atomic change. If the change affects features, architecture, or project structure, update README.md in the same commit.
