import asyncio, struct, json, time, os, socket, datetime
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from aiohttp import web
from zeroconf.asyncio import AsyncZeroconf
from zeroconf import ServiceInfo
import socket



# ============================================================
# GLOBAL BUFFERS
# ============================================================
# ================= CONFIG =================
UDP_PORT = 5006
HTTP_PORT = 8080
BUFFER_SIZE = 20000
FFT_SIZE = 2048
FFT_PERIOD = 0.25        # seconds
CLIENT_TIMEOUT = 5.0     # seconds (auto-disconnect)

# ============== CLIENT STORAGE ============
clients = {}   # client_id -> state
RAW_X, RAW_Y, RAW_Z, RAW_T = [], [], [], []
BUFFER_SIZE = 50000
WS_CLIENTS = set()
AZEROCONF = None


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


# ============== CLIENT STATE INIT =========
def new_client(cid):
    now = time.time()
    fname = f"vibration_client{cid}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    f = open(fname, "w", buffering=1)
    f.write("timestamp_us,x,y,z\n")
    print(f"[NEW CLIENT] id={cid} csv={fname}")

    return {
        "T": [], "X": [], "Y": [], "Z": [],
        "last_seen": now,
        "csv": f,
        "rpm": 0.0,
        "waterfall": []
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
        if len(data) != 11: return
        cid, ts, x, y, z = struct.unpack("<BIhhh", data)

        if cid not in clients:
            clients[cid] = new_client(cid)

        c = clients[cid]
        c["last_seen"] = time.time()

        c["T"].append(ts)
        c["X"].append(x)
        c["Y"].append(y)
        c["Z"].append(z)

        c["csv"].write(f"{ts},{x},{y},{z}\n")

        if len(c["T"]) > BUFFER_SIZE:
            for k in ("T","X","Y","Z"): c[k].pop(0)
        
        msg = {"type":"raw", "client_id":cid, "x":x, "y":y, "z":z}
        for ws in WS_CLIENTS:
            asyncio.create_task(ws.send_str(json.dumps(msg)))

# ============================================================
# AUTO SAMPLE RATE
# ============================================================

def auto_fs(timestamps):
    if len(timestamps) < 50: return 0
    d = np.diff(timestamps[-50:])
    d = d[d > 0]
    return 1e6 / np.median(d) if len(d) else 0


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
# ============== ANALYSIS LOOP ==============
async def analysis_task():
    while True:
        try:
            await asyncio.sleep(FFT_PERIOD)
            now = time.time()

            dead = [cid for cid,c in clients.items()
                    if now - c["last_seen"] > CLIENT_TIMEOUT]

            for cid in dead:
                print(f"[DISCONNECT] client {cid}")
                clients[cid]["csv"].close()
                del clients[cid]

            for cid,c in clients.items():
                if len(c["X"]) < FFT_SIZE: continue

                fs = auto_fs(c["T"])
                if fs <= 0: continue

                sx = np.array(c["X"][-FFT_SIZE:], float)
                sy = np.array(c["Y"][-FFT_SIZE:], float)
                sz = np.array(c["Z"][-FFT_SIZE:], float)

                sx = sx - np.mean(sx)
                sy = sy - np.mean(sx)
                sz = sz - np.mean(sx)

                
                rpm = auto_rpm(sx, fs)
                c["rpm"] = rpm

                fftx = np.abs(np.fft.rfft(sx))
                ffty = np.abs(np.fft.rfft(sy))
                fftz = np.abs(np.fft.rfft(sz))
                freqs = np.fft.rfftfreq(len(sx), 1/fs)

                pkt = {
                    "type": "fft",
                    "client": cid,
                    "rpm": rpm,
                    "freqs": freqs.tolist(),
                    "fft": fftx.tolist(),
                    "alive": True
                }

                for ws in WS_CLIENTS:
                    await ws.send_str(json.dumps(pkt))
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
    az = AsyncZeroconf()
    ip = socket.gethostbyname(socket.gethostname())
    info = ServiceInfo(
        "_vibration._udp.local.",
        "VibrationServer._vibration._udp.local.",
        addresses=[socket.inet_aton(ip)],
        port=UDP_PORT,
        server="vibration-server.local."
    )
    await az.async_register_service(info)
    print("[mDNS] vibration-server.local")

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
    await start_mdns()

    await loop.create_datagram_endpoint(lambda: UDPProtocol(), local_addr=("0.0.0.0", UDP_PORT))
    
    
    asyncio.create_task(analysis_task())

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/ws", websocket_handler)])

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", HTTP_PORT).start()

    print(f"UI → http://vibration-server.local:{HTTP_PORT}")
    while True: await asyncio.sleep(3600)

asyncio.run(main())
