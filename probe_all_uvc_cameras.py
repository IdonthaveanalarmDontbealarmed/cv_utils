# ===============================================================
# UVC Camera Probe — Fast-first Progressive + CSV (no CLI)
# ===============================================================

SETTINGS = {
    # Backends: Windows DSHOW=700, MSMF=1400, ANY=0 | Linux V4L2=200, GST=180, ANY=0
    "WIN_BACKENDS": [2300, 1900, 1800, 700, 1400, 2000, 2200, 2500, 2600, 0],
    "LIN_BACKENDS": [200, 180, 0],     # V4L2, GST, ANY

    # Windows enumeration fallback (probe indices 0..N)
    "MAX_INDEX_GUESS": 6,

    # Fast path, always tried first (quick, safe)
    "BASELINE_RES": [(640,480), (1280,720), (1920,1080)],     # low → mid → FHD
    "BASELINE_FMTS": ["YUY2", "MJPG"],                       # uncompressed → compressed

    # Expanded sweep (only if time left). Capped by MAX_WIDTH/HEIGHT.
    "SEARCH_FORMATS": ["YUY2", "GREY", "NV12", "MJPG", "H264"],   # uncompressed first
    "FORMAT_RANK":    {"YUY2":5, "GREY":4, "NV12":3, "MJPG":2, "H264":1},
    "SEARCH_RES": [
        (1920,1080), (1600,1200), (1440,1080), (1280,1024),
        (1280,800),  (1280,720),  (1024,768),  (800,600),
        (640, 512), (640,480),   
    ],
    "MAX_WIDTH": 2560, "MAX_HEIGHT": 1440,                   # hard cap for expanded sweep

    # Timing (keep it snappy)
    "OPEN_TIMEOUT_MS": 800,           # per (re)open
    "READ_TIMEOUT_MS": 800,           # per mode read window
    "WARMUP_FRAMES": 4,               # quick warmup
    "MIN_GOOD_FRAMES": 2,             # require >= non-blank frames to accept a mode
    "BLACK_STDDEV_THRESH": 2.5,       # below ⇒ treat as blank

    # Output
    "SAVE_ONE_PER_COMBO": True,       # save 1 frame for every successful combo
    "TOP_EXTRA_FRAMES": 2,            # extra frames for the chosen TOP mode only
    "JPEG_QUALITY": 92,

    # Budgets (so it never runs long)
    "WORKER_BUDGET_S": 12.0,          # internal per-device budget
    "PROC_TIMEOUT_S": 14.0            # outer hard timeout
}

# ===============================================================
import os, sys, re, glob, csv, time, shutil, subprocess, platform

# Silence noisy OpenCV plugins (e.g., OBSENSOR)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

def is_win(): return platform.system().lower().startswith("win")
def is_linux(): return platform.system().lower()=="linux"

BACKEND_NAMES={700:"DSHOW",1400:"MSMF",200:"V4L2",190:"V4L",180:"GSTREAMER",0:"ANY"}

# ---------- Helpers ----------
def list_available_backends():
    try:
        import cv2
        from cv2 import videoio_registry as vir
        ids = getattr(vir, "getBackends", lambda: [])()
        if ids:
            names = [getattr(vir, "getBackendName", lambda x:str(x))(i) for i in ids]
            return list(zip(names, ids))
    except Exception:
        pass
    try:
        import cv2
        return [("see cv2.getBuildInformation()", -1)]
    except Exception:
        return []

def enumerate_linux():
    devs=sorted(glob.glob("/dev/video[0-9]*"), key=lambda p:int(re.search(r"(\d+)$",p).group(1)))
    return devs

def enumerate_windows(max_guess:int):
    try:
        import cv2
        found=[]
        for i in range(max_guess+1):
            cap=cv2.VideoCapture(i)
            ok=cap.isOpened()
            try: cap.release()
            except: pass
            if ok: found.append(i)
        return found if found else [0]
    except Exception:
        return [0]

def channel_of(dev):
    if is_linux():
        m=re.search(r"(\d+)$", dev); return int(m.group(1)) if m else 0
    return int(dev)

# ---------- Worker (fast-first + capped sweep) ----------
def worker(device, backend, outdir, manifest_path, S):
    import cv2, numpy as np, time
    start=time.time()
    def time_left(): return S["WORKER_BUDGET_S"] - (time.time()-start)
    def fourcc_int(tag): 
        try: return cv2.VideoWriter_fourcc(*tag) if tag else 0
        except: return 0
    def fourcc_tag_from_cap(cap):
        try:
            v=int(cap.get(cv2.CAP_PROP_FOURCC))
            if v==0: return "AUTO"
            return "".join([chr((v>>(8*i))&0xFF) for i in range(4)])
        except: return "AUTO"
    def non_blank(frm):
        g=cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY); s=float(g.std()); return (s>=S["BLACK_STDDEV_THRESH"], s)

    dev = int(device) if is_win() else device
    ch  = channel_of(device) if is_linux() else int(device)

    def reopen():
        cap=cv2.VideoCapture(dev, backend)
        t0=time.time()
        while not cap.isOpened() and (time.time()-t0)*1000 < S["OPEN_TIMEOUT_MS"]:
            time.sleep(0.01)
        return cap if cap.isOpened() else None

    def try_mode(fmt, W, H, need_frames):
        if time_left() <= 0: return None
        cap = reopen()
        if cap is None: return None
        # Request FOURCC & resolution (fmt="" means don't touch FOURCC)
        if fmt:
            try: cap.set(cv2.CAP_PROP_FOURCC, fourcc_int(fmt))
            except: pass
        if W and H:
            try: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
            except: pass
            try: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            except: pass
        # warmup
        for _ in range(S["WARMUP_FRAMES"]):
            ret,_=cap.read()
            if not ret: break
        # collect
        good=[]; t1=time.time()
        while (time.time()-t1)*1000 < S["READ_TIMEOUT_MS"] and len(good) < need_frames:
            ret,frm=cap.read()
            if not ret or frm is None: time.sleep(0.005); continue
            ok,stdv=non_blank(frm)
            if ok: good.append((frm,stdv))
        # reported mode
        try:
            aw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); ah=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except:
            aw,ah=W,H
        rep_fmt = fourcc_tag_from_cap(cap)
        try: cap.release()
        except: pass
        if len(good) < S["MIN_GOOD_FRAMES"]: return None
        return {"fmt_req":fmt or "AUTO","fmt_rep":rep_fmt,"reqW":W or -1,"reqH":H or -1,"actW":aw,"actH":ah,"area":aw*ah,"frames":good}

    successes=[]

    # Phase 0: super-fast default read (no format/res requests)
    res = try_mode(fmt="", W=None, H=None, need_frames=S["MIN_GOOD_FRAMES"])
    if res: successes.append(res)

    # Phase 1: baseline (low→mid→FHD with YUY2 then MJPG)
    for W,H in S["BASELINE_RES"]:
        for fmt in S["BASELINE_FMTS"]:
            if time_left() <= 0: break
            res = try_mode(fmt,W,H,need_frames=S["MIN_GOOD_FRAMES"])
            if res: successes.append(res)

    # Phase 2: capped sweep (≤ MAX_WIDTH/HEIGHT)
    tested = {(r["fmt_req"], r["reqW"], r["reqH"]) for r in successes}
    for fmt in S["SEARCH_FORMATS"]:
        for (W,H) in S["SEARCH_RES"]:
            if W>S["MAX_WIDTH"] or H>S["MAX_HEIGHT"]: continue
            key=(fmt,W,H)
            if key in tested: continue
            if time_left() <= 0: break
            res = try_mode(fmt,W,H,need_frames=S["MIN_GOOD_FRAMES"])
            if res: successes.append(res)

    if not successes: return 3

    # Rank: uncompressed first, then area, then avg stddev
    def avgstd(r): return sum(s for _,s in r["frames"])/len(r["frames"])
    for r in successes:
        r["rank"]=SETTINGS["FORMAT_RANK"].get(r["fmt_req"],0)
        r["avgstd"]=avgstd(r)
    successes.sort(key=lambda r:(r["rank"], r["area"], r["avgstd"]), reverse=True)
    top = successes[0]

    # Save & manifest
    with open(manifest_path,"a",newline="") as mf:
        wr=csv.writer(mf)
        # one frame for every success (if enabled)
        if SETTINGS["SAVE_ONE_PER_COMBO"]:
            for r in successes:
                frm,stdv = r["frames"][0]
                bname = BACKEND_NAMES.get(backend,str(backend))
                fname = f"probe_{ch:01d}_{bname}_{r['fmt_req']}_{r['actW']}x{r['actH']}_f1.jpg"
                path  = os.path.join(outdir,fname)
                cv2.imwrite(path, frm, [int(cv2.IMWRITE_JPEG_QUALITY), SETTINGS["JPEG_QUALITY"]])
                wr.writerow([ch, backend, bname, r["fmt_req"], r["fmt_rep"],
                             f"{r['reqW']}x{r['reqH']}", r["actW"], r["actH"],
                             1, round(r['frames'][0][1],2), False, False, fname])
        # extra frames for TOP
        extra = min(SETTINGS["TOP_EXTRA_FRAMES"], max(0, len(top["frames"])-1))
        bname = BACKEND_NAMES.get(backend,str(backend))
        for i in range(extra):
            frm,stdv = top["frames"][i+1]
            fname = f"probe_{ch:03d}_{bname}_{top['fmt_req']}_{top['actW']}x{top['actH']}_f{i+2}.jpg"
            path  = os.path.join(outdir,fname)
            cv2.imwrite(path, frm, [int(cv2.IMWRITE_JPEG_QUALITY), SETTINGS["JPEG_QUALITY"]])
            wr.writerow([ch, backend, bname, top["fmt_req"], top["fmt_rep"],
                         f"{top['reqW']}x{top['reqH']}", top["actW"], top["actH"],
                         i+2, round(stdv,2), False, True, fname])
    return 0

# ---------- Main ----------
def main():
    print("=== UVC Probe (Fast-first) ===")
    print(f"OS: {platform.system()} | Python {platform.python_version()}")

    # Show available backends
    bks=list_available_backends()
    if bks:
        print("[info] OpenCV backends available: " + ", ".join(f"{n}({c})" for n,c in bks))
    else:
        print("[info] OpenCV backends: unknown; proceeding")

    # Reset output
    if os.path.exists("probe"):
        try: shutil.rmtree("probe"); print("[info] removed old ./probe")
        except Exception as e: print(f"[warn] rm probe: {e}")
    os.makedirs("probe", exist_ok=True)

    # Manifest
    manifest=os.path.join("probe","manifest.csv")
    with open(manifest,"w",newline="") as mf:
        csv.writer(mf).writerow(
            ["device_channel","backend_code","backend_name",
             "format_requested","format_reported","requested_resolution",
             "actual_width","actual_height","frame_index","stddev","is_black","is_top","filename"]
        )

    # Enumerate
    if is_linux():
        candidates=enumerate_linux(); backends=SETTINGS["LIN_BACKENDS"]
    else:
        candidates=enumerate_windows(SETTINGS["MAX_INDEX_GUESS"]); backends=SETTINGS["WIN_BACKENDS"]

    if not candidates:
        print("[result] no devices found."); return
    print(f"[info] candidates: {', '.join(map(str,candidates))}")

    successes=0; fails=[]
    for dev in candidates:
        ch=channel_of(dev) if is_linux() else int(dev)
        print(f"\n[probe] channel {ch:03d} ({dev})")
        got=False
        for bk in backends:
            cmd=[sys.executable, os.path.abspath(__file__), "--worker", str(dev), str(bk), "probe", manifest]
            bname=BACKEND_NAMES.get(bk,str(bk))
            print(f"  [try] backend={bname} ...", end="", flush=True)
            t0=time.time()
            try:
                r=subprocess.run(cmd, text=True, capture_output=True, timeout=SETTINGS["PROC_TIMEOUT_S"])
                print(f" done ({int((time.time()-t0)*1000)} ms)")
                if r.returncode==0:
                    got=True; successes+=1; print(f"  [ok] backend={bname}"); break
                else:
                    print(f"  [fail] rc={r.returncode}")
            except subprocess.TimeoutExpired:
                print(" timeout; killed")
            pass
                print("  [passed] worker subrpocess yielded a frame")
        if not got:
            fails.append(ch); print(f"  [skip] channel {ch:03d} failed on all backends")

    print("\n=== Summary ===")
    print(f"[result] devices OK: {successes} | failed: {len(fails)}")
    if fails: print("[failed] " + ", ".join(f"{c:03d}" for c in fails))
    print("Manifest: probe/manifest.csv")

# ---------- Worker entry ----------
if len(sys.argv)==6 and sys.argv[1]=="--worker":
    sys.exit(worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], SETTINGS))
else:
    main()
