# Plot ratings per video

This small script reads a CSV (default `db.csv`) and creates one PNG plot per unique video id.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the script (from project folder):

```bash
python script.py --csv db.csv --outdir plots
```

- Output: a `plots/` directory with one PNG per video id (file names are sanitized).
- Use `--show` to display plots interactively (requires a GUI-capable Python/matplotlib backend).
