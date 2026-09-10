"""Общие константы протокола RoboOM.

ВАЖНО: значения должны совпадать с Android-приложением
(android/app/src/main/java/com/roboom/teleop/model/Protocol.kt).
"""

SVC_NAME = "roboom"
PROTO_VERSION = 1

# UDP-порты
BEACON_PORT = 50101   # ПК рассылает broadcast-маячок "я здесь"
DATA_PORT = 50103     # подписка телефона, кадры позиций, обратная связь

# Порядок суставов SO-101 — во всех массивах кадров именно такой
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]

# BLE (телефон выступает GATT-сервером, ПК подключается как клиент)
BLE_SERVICE_UUID = "8e5c0001-5b1e-4f6a-93f5-4b0c9d0e8a01"
BLE_CHAR_FRAME = "8e5c0002-5b1e-4f6a-93f5-4b0c9d0e8a01"   # ПК -> телефон: кадры (write w/o response)
BLE_CHAR_CONFIG = "8e5c0003-5b1e-4f6a-93f5-4b0c9d0e8a01"  # ПК -> телефон: калибровка ведомой (чанки)
BLE_CHAR_STATE = "8e5c0004-5b1e-4f6a-93f5-4b0c9d0e8a01"   # телефон -> ПК: позиции ведомой (notify)
BLE_FRAME_MAGIC = 0xA5
