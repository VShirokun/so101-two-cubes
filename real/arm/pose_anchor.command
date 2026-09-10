#!/bin/bash
cd "$(dirname "$0")"
exec /opt/anaconda3/envs/lerobot/bin/python -u pose_anchor.py 2>&1 | tee /tmp/roboom_arm/anchor.log
