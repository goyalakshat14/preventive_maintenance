import asyncio
import json
import struct
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from aiohttp import web
from zeroconf.asyncio import AsyncZeroconf
from zeroconf import ServiceInfo
import socket



# ============================================================
# GLOBAL BUFFERS
# ============================================================
RAW_X, RAW_Y, RAW_Z, RAW_T = [], [], [], []
BUFFER_SIZE = 50000
WS_CLIENTS = set()
AZEROCONF = None
client_nodes = {}

# ============================================================
# CONFIG (all auto-updated)
# ============================================================
CONFIG = {
    "sample_rate": 0,          # Auto detected
    "fft_size": 2048,
    "fft_update_ms": 200,
    "filter_type": "none",     # none / low / high / band
    "low_cut": 5,
    "high_cut": 50,
    "band_low": 100,
    "band_high": 500,
    "rpm": 0                   # Auto detected
}

# ============================================================
# BEARING GEOMETRY (replace values for your bearing)
# ============================================================
BALL_DIAMETER = 3.5
PITCH_DIAMETER = 25.0
CONTACT_ANGLE = 0
BALL_COUNT = 8

some_error = 0
def bearing_freqs():
    if CONFIG["rpm"] <= 0:
        return {"BPFO":0, "BPFI":0, "BSF":0, "FT":0}

    fr = CONFIG["rpm"] / 60
    BD = BALL_DIAMETER
    PD = PITCH_DIAMETER
    ca = np.cos(np.radians(CONTACT_ANGLE))

    return {
        "BPFO": (BALL_COUNT/2) * fr * (1 - (BD/PD)*ca),
        "BPFI": (BALL_COUNT/2) * fr * (1 + (BD/PD)*ca),
        "BSF":  (PD/(2*BD)) * fr * (1 - (BD**2/PD**2)*ca),
        "FT":   0.5 * fr * (1 - (BD/PD)*ca)
    }



import datetime

# ============================================================
# CSV LOGGING SETUP WITH AUTO-ROLLOVER
# ============================================================
csv_file = None
csv_path = None
csv_start_time = None
csv_max_size = 10_000_000     # 10 MB
csv_time_limit = 3600         # 1 hour (in seconds)

import datetime
import os

def open_new_csv():
    """Create a new CSV file."""
    global csv_file, csv_path, csv_start_time
    if csv_file:
        csv_file.close()

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    csv_path = f"vibration_raw_{timestamp}.csv"
    csv_file = open(csv_path, "w", buffering=1)
    csv_file.write("timestamp_us,x,y,z\n")
    csv_start_time = now

    print(f"[CSV] Started new log file: {csv_path}")

def check_rollover():
    """Check if file size or time limit exceeded."""
    if not csv_file:
        return

    # Size-based rollover
    if os.path.getsize(csv_path) >= csv_max_size:
        print("[CSV] Size limit exceeded. Rolling over...")
        open_new_csv()
        return

    # Time-based rollover
    elapsed = (datetime.datetime.now() - csv_start_time).total_seconds()
    if elapsed >= csv_time_limit:
        print("[CSV] Time limit exceeded. Rolling over...")
        open_new_csv()
        return



# ============================================================
# FILTERING FUNCTIONS
# ============================================================
def apply_filter(signal):
    if CONFIG["sample_rate"] <= 0:
        return signal

    fs = CONFIG["sample_rate"]
    nyq = fs / 2

    if CONFIG["filter_type"] == "low":
        b, a = butter(4, CONFIG["low_cut"]/nyq, "low")
        return filtfilt(b, a, signal)

    if CONFIG["filter_type"] == "high":
        b, a = butter(4, CONFIG["high_cut"]/nyq, "high")
        return filtfilt(b, a, signal)

    if CONFIG["filter_type"] == "band":
        b, a = butter(4, [CONFIG["band_low"]/nyq, CONFIG["band_high"]/nyq], "band")
        return filtfilt(b, a, signal)

    return signal

def envelope(signal):
    analytic = hilbert(signal)
    return np.abs(analytic)

# ============================================================
# UDP RECEIVER
# ============================================================
class UDPProtocol(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        print("UDP ready.")
        self.transport = transport

    def datagram_received(self, data, addr):
        if len(data) != 11:
            return

        client_id, ts, x, y, z = struct.unpack("<BIhhh", data)

        if csv_file:
            csv_file.write(f"{ts},{x},{y},{z}\n")
            check_rollover()

        c = client_nodes[client_id]
        c["RAW_T"].append(ts)
        c["RAW_X"].append(x)
        c["RAW_Y"].append(y)
        c["RAW_Z"].append(z)

        if len(c["RAW_T"]) > BUFFER_SIZE:
            c["RAW_T"].pop(0)
            c["RAW_X"].pop(0)
            c["RAW_Y"].pop(0)
            c["RAW_Z"].pop(0)
        # RAW_T.append(ts)
        # RAW_X.append(x)
        # RAW_Y.append(y)
        # RAW_Z.append(z)

        # if len(RAW_T) > BUFFER_SIZE:
        #     RAW_T.pop(0); RAW_X.pop(0); RAW_Y.pop(0); RAW_Z.pop(0)

        # Live waveforms
        msg = {"type":"raw", "client_id":client_id, "x":x, "y":y, "z":z}
        for ws in WS_CLIENTS:
            asyncio.create_task(ws.send_str(json.dumps(msg)))

# ============================================================
# AUTO SAMPLE RATE
# ============================================================
def auto_sample_rate():
    if len(RAW_T) < 200:
        return 0

    diffs = np.diff(RAW_T[-200:])
    diffs = diffs[diffs > 0]   # remove timestamp rollovers
    if len(diffs) == 0:
        return 0

    median_us = np.median(diffs)
    return 1_000_000.0 / median_us

# ============================================================
# AUTO RPM DETECTION
# ============================================================
def auto_rpm(signal, fs):
    sig = signal - np.mean(signal)

    # Autocorrelation RPM
    corr = np.correlate(sig, sig, mode="full")
    corr = corr[len(corr)//2:]

    min_i = int(fs * 0.01)
    max_i = int(fs * 0.2)

    if max_i >= len(corr):
        return 0

    seg = corr[min_i:max_i]
    peak = np.argmax(seg) + min_i

    rpm_auto = 60 * fs / peak
    
    # FFT RPM
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))
    low = freqs < 1000

    if np.sum(low) == 0:
        return rpm_auto

    peak_f = freqs[low][np.argmax(fft_vals[low])]
    rpm_fft = peak_f * 60
    return rpm_fft

# ============================================================
# FFT TASK
# ============================================================
async def fft_task():
    waterfall = []
    global some_error
    global WS_CLIENTS

    while True:
        try:
            temp_ws_clients = WS_CLIENTS.copy()
            
            await asyncio.sleep(CONFIG["fft_update_ms"]/1000)

            if len(RAW_X) < CONFIG["fft_size"]:
                continue

            CONFIG["sample_rate"] = auto_sample_rate()

            fs = CONFIG["sample_rate"]
            if fs <= 0:
                continue
            # print("getting data")
            sx = np.array(RAW_X[-CONFIG["fft_size"]:], float)
            sy = np.array(RAW_Y[-CONFIG["fft_size"]:], float)
            sz = np.array(RAW_Z[-CONFIG["fft_size"]:], float)

            sx = sx - np.mean(sx)
            sy = sy - np.mean(sx)
            sz = sz - np.mean(sx)
            # Filtering
            sx_f = apply_filter(sx)
            sy_f = apply_filter(sy)
            sz_f = apply_filter(sz)

            # Envelope FFT (for bearing)
            env = envelope(sx_f)

            # Auto RPM detect
            CONFIG["rpm"] = auto_rpm(sx_f, fs)

            # FFTs
            freqs = np.fft.rfftfreq(CONFIG["fft_size"], 1/fs)
            fft_vals = np.abs(np.fft.rfft(sx_f))
            fx = np.abs(np.fft.rfft(sx_f))
            fy = np.abs(np.fft.rfft(sy_f))
            fz = np.abs(np.fft.rfft(sz_f))
            fe = np.abs(np.fft.rfft(env))

            # Waterfall (X only)
            waterfall.append(fx.tolist())
            if len(waterfall) > 80:
                waterfall.pop(0)

            packet = {
                "type":"fft",
                "freqs": freqs.tolist(),
                "fft": fft_vals.tolist(),
                "envelope": fe.tolist(),
                "rpm": CONFIG["rpm"],
                "bearing": {},
                "waterfall": waterfall,
                "fs":fs
            }

            for ws in temp_ws_clients:
                try:
                    await ws.send_str(json.dumps(packet))
                except Exception as inst:
                    print(inst)
        except Exception as inst:
            print(inst)


# ============================================================
# WEBSOCKET HANDLER
# ============================================================
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    WS_CLIENTS.add(ws)

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            cfg = json.loads(msg.data)
            if cfg["cmd"] == "config":
                for k,v in cfg["data"].items():
                    CONFIG[k] = v

    WS_CLIENTS.remove(ws)
    return ws


async def start_mdns():
    global AZEROCONF
    ip = socket.gethostbyname(socket.gethostname())
    info = ServiceInfo(
        "_vibration._udp.local.",
        "VibrationServer._vibration._udp.local.",
        addresses=[socket.inet_aton(ip)],
        port=5006,
        properties={},
        server="vibration-server.local."
    )
    AZEROCONF = AsyncZeroconf()
    await AZEROCONF.async_register_service(info)
    print(f"[mDNS] Server advertised as vibration-server.local")


# ============================================================
# HTTP ROUTES
# ============================================================
async def index(request):
    return web.FileResponse("index.html")

# ============================================================
# MAIN
# ============================================================
async def main():
    loop = asyncio.get_running_loop()
    open_new_csv()
    await start_mdns()

    await loop.create_datagram_endpoint(lambda: UDPProtocol(), local_addr=("0.0.0.0", 5006))
    
    
    asyncio.create_task(fft_task())

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/ws", websocket_handler)])

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", 8080).start()

    print("Server running at http://localhost:8080/")
    while True:
        await asyncio.sleep(3600)

asyncio.run(main())
