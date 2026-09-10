#!/bin/bash
cd "$(dirname "$0")"
exec /opt/anaconda3/envs/lerobot/bin/python - <<'PY' > /tmp/roboom_cam/probe.log 2>&1
import cv2, time
try:
    import AVFoundation as AV
    devs = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)
    for k, d in enumerate(devs):
        print(f"AV[{k}]: {d.localizedName()}  uid={d.uniqueID()}")
except Exception as e:
    print("pyobjc:", e)
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"index {i}: не открылась")
        continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    time.sleep(0.3)
    ok, fr = False, None
    for _ in range(8):
        ok, fr = cap.read()
    if ok:
        print(f"index {i}: кадр {fr.shape[1]}x{fr.shape[0]}")
        cv2.imwrite(f"/tmp/roboom_cam/probe_{i}.jpg", fr)
    else:
        print(f"index {i}: открылась, кадра нет")
    cap.release()
print("probe done")
PY
