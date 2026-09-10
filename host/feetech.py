"""Минимальная реализация протокола Feetech STS3215 (SO-101) поверх pyserial.

Не зависит от lerobot — только pyserial. Little-endian, пакеты вида:
  TX: FF FF ID LEN INSTR PARAMS.. CHK,  CHK = ~(ID+LEN+INSTR+sum(PARAMS)) & 0xFF
  RX: FF FF ID LEN ERR  PARAMS.. CHK
"""
from __future__ import annotations

import glob
import time

import serial

# Инструкции
PING, READ, WRITE = 0x01, 0x02, 0x03
SYNC_READ, SYNC_WRITE = 0x82, 0x83
BROADCAST_ID = 0xFE

# Регистры STS3215
ADDR_MODEL = 3                 # u16
ADDR_MIN_POSITION = 9          # u16
ADDR_MAX_POSITION = 11         # u16
ADDR_TORQUE_ENABLE = 40        # u8
ADDR_ACCELERATION = 41         # u8
ADDR_GOAL_POSITION = 42        # u16
ADDR_PRESENT_POSITION = 56     # u16, знак в бите 15
ADDR_PRESENT_VOLTAGE = 62      # u8, 0.1 В
ADDR_PRESENT_TEMPERATURE = 63  # u8, °C

MODEL_NAMES = {777: "STS3215", 2825: "STS3250"}


def _checksum(core: bytes) -> int:
    return (~sum(core)) & 0xFF


def decode_signed(value: int, sign_bit: int = 15) -> int:
    """Feetech кодирует отрицательные числа знаковым битом (не two's complement)."""
    if value & (1 << sign_bit):
        return -(value & ((1 << sign_bit) - 1))
    return value


def find_ports() -> list[str]:
    """Кандидаты последовательных портов адаптера сервоприводов на macOS/Linux."""
    pats = ["/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*",
            "/dev/cu.SLAB*", "/dev/ttyACM*", "/dev/ttyUSB*"]
    out: list[str] = []
    for p in pats:
        out.extend(sorted(glob.glob(p)))
    return out


class FeetechBus:
    def __init__(self, port: str, baud: int = 1_000_000, timeout: float = 0.02):
        self.port_name = port
        self.ser = serial.Serial(port, baud, timeout=timeout, write_timeout=0.5)
        self.sync_read_supported = True  # выключается автоматически при первом провале

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

    # --- низкий уровень -------------------------------------------------
    def _send(self, sid: int, instr: int, params: bytes = b"") -> None:
        core = bytes([sid, len(params) + 2, instr]) + params
        self.ser.reset_input_buffer()
        pkt = b"\xFF\xFF" + core + bytes([_checksum(core)])
        try:
            self.ser.write(pkt)
        except serial.SerialTimeoutException:
            # pyserial считает таймаут по часам процесса: при нехватке CPU
            # (load average 84 при нарезке датасета, 09.09.2026) он срабатывает,
            # хотя байты уже ушли. Один повтор вместо падения серии.
            time.sleep(0.005)
            self.ser.reset_output_buffer()
            self.ser.write(pkt)

    def _read_status(self, deadline: float) -> tuple[int, int, bytes] | None:
        """Один статус-пакет: (id, err, params) либо None по таймауту."""
        buf = b""
        while time.monotonic() < deadline:
            chunk = self.ser.read(max(1, 6 - len(buf)))
            if chunk:
                buf += chunk
            i = buf.find(b"\xFF\xFF")
            if i > 0:
                buf = buf[i:]
            elif i < 0 and len(buf) > 1:
                buf = buf[-1:]
            if len(buf) >= 4:
                total = 4 + buf[3]
                if len(buf) >= total:
                    pkt, buf = buf[:total], buf[total:]
                    core = pkt[2:total - 1]
                    if _checksum(core) == pkt[total - 1]:
                        return pkt[2], pkt[4], pkt[5:total - 1]
                    buf = pkt[2:] + buf  # битый пакет — ресинхронизация
        return None

    # --- базовые операции -------------------------------------------------
    def ping(self, sid: int, timeout: float = 0.05) -> bool:
        self._send(sid, PING)
        r = self._read_status(time.monotonic() + timeout)
        return r is not None and r[0] == sid

    def read(self, sid: int, addr: int, size: int, timeout: float = 0.05) -> bytes | None:
        self._send(sid, READ, bytes([addr, size]))
        r = self._read_status(time.monotonic() + timeout)
        if r and r[0] == sid and len(r[2]) == size:
            return r[2]
        return None

    def read_u8(self, sid: int, addr: int) -> int | None:
        b = self.read(sid, addr, 1)
        return b[0] if b else None

    def read_u16(self, sid: int, addr: int, signed: bool = False) -> int | None:
        b = self.read(sid, addr, 2)
        if not b:
            return None
        v = b[0] | (b[1] << 8)
        return decode_signed(v) if signed else v

    def write(self, sid: int, addr: int, data: bytes, expect_status: bool = True) -> None:
        self._send(sid, WRITE, bytes([addr]) + data)
        if expect_status and sid != BROADCAST_ID:
            self._read_status(time.monotonic() + 0.02)  # статус читаем и игнорируем

    def write_u8(self, sid: int, addr: int, value: int) -> None:
        self.write(sid, addr, bytes([value & 0xFF]))

    def write_u16(self, sid: int, addr: int, value: int) -> None:
        v = max(0, min(0xFFFF, int(value)))
        self.write(sid, addr, bytes([v & 0xFF, (v >> 8) & 0xFF]))

    # --- групповые операции ------------------------------------------------
    def sync_write(self, addr: int, size: int, values: dict[int, int]) -> None:
        params = bytes([addr, size])
        for sid, v in values.items():
            v = max(0, min((1 << (8 * size)) - 1, int(v)))
            data = bytes((v >> (8 * k)) & 0xFF for k in range(size))
            params += bytes([sid]) + data
        self._send(BROADCAST_ID, SYNC_WRITE, params)

    def read_positions(self, ids: list[int]) -> dict[int, int]:
        """Позиции всех сервоприводов; сначала SYNC_READ, при неудаче — по одному."""
        out: dict[int, int] = {}
        if self.sync_read_supported:
            self._send(BROADCAST_ID, SYNC_READ,
                       bytes([ADDR_PRESENT_POSITION, 2]) + bytes(ids))
            deadline = time.monotonic() + 0.02 + 0.01 * len(ids)
            for _ in ids:
                r = self._read_status(deadline)
                if r and len(r[2]) == 2:
                    out[r[0]] = decode_signed(r[2][0] | (r[2][1] << 8))
            if out:
                return out
            self.sync_read_supported = False  # прошивка без SYNC_READ
        for sid in ids:
            v = self.read_u16(sid, ADDR_PRESENT_POSITION, signed=True)
            if v is not None:
                out[sid] = v
        return out

    # --- удобные обёртки -----------------------------------------------------
    def scan(self, id_range=range(1, 11)) -> list[int]:
        return [i for i in id_range if self.ping(i)]

    def set_torque(self, ids: list[int], on: bool) -> None:
        self.sync_write(ADDR_TORQUE_ENABLE, 1, {i: 1 if on else 0 for i in ids})

    def set_acceleration(self, ids: list[int], acc: int) -> None:
        self.sync_write(ADDR_ACCELERATION, 1, {i: acc for i in ids})

    def write_goals(self, goals: dict[int, int]) -> None:
        self.sync_write(ADDR_GOAL_POSITION, 2, goals)
