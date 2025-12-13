# Plot ratings per video

This small script reads a CSV (default `db.csv`) and creates one PNG plot per unique video id.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the script (from project folder):

```bash
python script.py (for all users)
python script.py --user uuid (for one user)
python script.py --videodir videos (overlay first video frame per subplot)
```

Place the corresponding `.mp4` files in the `videos/` directory (or point `--videodir` elsewhere) to embed each video's first frame inside its rating subplot. Install requirements with `python -m pip install -r requirements.txt`.

