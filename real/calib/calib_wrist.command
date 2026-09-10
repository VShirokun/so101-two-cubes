#!/bin/bash
cd "$(dirname "$0")"
exec /opt/anaconda3/envs/lerobot/bin/python -u intrinsics_from_markers.py --dir /tmp/roboom_wrist --out wrist --f-expect 0 --views 50 2>&1 | tee /tmp/roboom_wrist/calib.log
