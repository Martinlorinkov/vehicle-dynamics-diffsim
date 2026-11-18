# downloads track centerlines from the TUM racetrack database
# (github.com/TUMFTM/racetrack-database, LGPL-3.0). not committed to this repo,
# run this once:  python data/get_tracks.py

import os
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/{}.csv"
TRACKS = ["Nuerburgring", "Spa", "Monza", "Silverstone"]

here = os.path.dirname(os.path.abspath(__file__))
names = sys.argv[1:] if len(sys.argv) > 1 else TRACKS

for name in names:
    out = os.path.join(here, name + ".csv")
    if os.path.exists(out):
        print(f"{name}: already have it")
        continue
    urllib.request.urlretrieve(URL.format(name), out)
    print(f"{name}: saved to {out}")
