#!/bin/bash
cd "$(dirname "$0")"
exec /opt/anaconda3/envs/lerobot/bin/python cam_viewer.py --dir /tmp/roboom_wrist --title "RoboOM wrist cam"
