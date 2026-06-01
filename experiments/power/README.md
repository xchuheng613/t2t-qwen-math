# Power Usage Logs

This folder keeps the archived power/cost monitor logs from the final project
run period.

New power monitor runs write to the ignored `results/` folder by default:

```bash
python scripts/power_cost_monitor.py --label my_run -- python your_command.py
```

Move only final summaries here if they need to be retained in git.
