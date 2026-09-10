#!/bin/bash
cd "$(dirname "$0")"
exec /opt/anaconda3/envs/lerobot/bin/python -u intrinsics_from_markers.py --dir /tmp/roboom_cam --out c920 2>&1 | tee /tmp/roboom_cam/calib.log
