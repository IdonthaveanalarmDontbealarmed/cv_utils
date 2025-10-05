# uvc_quad_byname.py — use Windows camera NAMES; resolve CV2 indices automatically
import os, random, string
import time, threading, ctypes, subprocess, re, shutil
from dataclasses import dataclass
from queue import Queue, Empty
from typing import List, Optional, Tuple
import numpy as np, cv2

# Optional deps
try:
    import av; HAVE_AV=True
except: HAVE_AV=False
try:
    import vlc; HAVE_VLC=True
except: HAVE_VLC=False
try:
    from pygrabber.dshow_graph import FilterGraph
    HAVE_PYGRABBER=True
except: HAVE_PYGRABBER=False

# ---------- Tunables ----------
MAX_QUEUE=2
WINDOW="uvc-quad"
TILE_H=480
FONT=cv2.FONT_HERSHEY_SIMPLEX
DISPLAY_FPS=60
STALE_OK_MS=2000
CAPTURE_DIR="CAPTURE"
AV_PIXELFMT_MAP={"MJPG":"mjpeg","YUY2":"yuyv422","":""}
CV2_BK_MAP={"DSHOW":cv2.CAP_DSHOW,"MSMF":cv2.CAP_MSMF,"ANY":0,"NONE":None}
VLC_COMMON="--quiet --no-audio --no-video-title-show --no-osd --no-xlib --vout=vmem"

# ---------- Device enumeration / name→index (DirectShow) ----------
def list_dshow_names() -> List[str]:
    if HAVE_PYGRABBER:
        try:
            g=FilterGraph()
            return g.get_input_devices()
        except: pass
    if shutil.which("ffmpeg"):
        try:
            out = subprocess.run(
                ["ffmpeg","-hide_banner","-list_devices","true","-f","dshow","-i","dummy"],
                capture_output=True, text=True, check=False
            ).stderr
            names=[]; in_vid=False
            for line in out.splitlines():
                if "DirectShow video devices" in line: in_vid=True; continue
                if "DirectShow audio devices" in line: in_vid=False; continue
                if in_vid:
                    m=re.search(r'\"(.+?)\"', line)
                    if m: names.append(m.group(1))
            return names
        except: pass
    return []

def resolve_dshow_index_by_name(name:str) -> Optional[int]:
    names = list_dshow_names()
    if not names: return None
    low=[n.lower() for n in names]; target=name.lower()
    if target in low: return low.index(target)
    hits=[i for i,n in enumerate(low) if target in n]
    return hits[0] if hits else None

# ---------- Spec ----------
@dataclass
class CamSpec:
    title:str
    device_name:str
    index_hint:int              # fallback index when name→index fails
    fourcc:str                  # "" for default
    size:Tuple[int,int]         # (0,0) for default
    fps:int                     # 0 for default
    backends:List[str]          # e.g. ["CV2","AV","VLC"]
    cv2_bks:List[str]           # e.g. ["NONE","DSHOW","ANY"]

# ----- Your names -----
CAMS:List[CamSpec]=[
    CamSpec("A","AFN_Cap video",            0, "",      (0,0),      0,  ["CV2","AV","VLC"], ["NONE","DSHOW","ANY"]),
    CamSpec("B","kit0",                     1, "",      (0,0),      0,  ["CV2","AV","VLC"], ["NONE","DSHOW","ANY"]),
    CamSpec("C","USB2.0 FHD UVC WebCam",    2, "",      (0,0),      0,  ["CV2","AV","VLC"], ["NONE","DSHOW","ANY"]),
    CamSpec("D","UVC Camera",               3, "MJPG",  (1920,1080),30, ["CV2","AV","VLC"], ["DSHOW","ANY"]),
]
# To use defaults: size=(0,0), fps=0, fourcc=""

class FrameGrabber:
    def __init__(self, s:CamSpec):
        self.s=s; self.q=Queue(MAX_QUEUE)
        self._t=None; self._stop=False; self.err=""
        self._av_c=None; self._cv2_cap=None
        self._vlc_inst=None; self._vlc_player=None; self._vlc_media=None
        self._vlc_buf=None; self._vlc_cb_set=False
        self._lock=threading.Lock()
        self._last_img=None; self._last_ts=0.0
        self._cv2_bk_used=""; self._cv2_idx_used=None

    def start(self):
        if self._t and self._t.is_alive(): return
        self._stop=False; self._t=threading.Thread(target=self._run,daemon=True); self._t.start()

    def stop(self):
        self._stop=True
        if self._t: self._t.join(timeout=1.0)
        with self._lock: self._close_all()

    def _close_all(self):
        try:
            if self._av_c: self._av_c.close()
        except: pass
        self._av_c=None
        try:
            if self._cv2_cap: self._cv2_cap.release()
        except: pass
        self._cv2_cap=None
        try:
            if self._vlc_player: self._vlc_player.stop(); self._vlc_player.release()
        except: pass
        self._vlc_player=None; self._vlc_media=None; self._vlc_inst=None
        self._vlc_buf=None; self._vlc_cb_set=False

    def _run(self):
        while not self._stop:
            opened=False
            for bk in self.s.backends:
                if self._stop: break
                try:
                    if bk=="AV" and HAVE_AV: opened=self._open_av_by_name()
                    elif bk=="CV2": opened=self._open_cv2_by_name()
                    elif bk=="VLC" and HAVE_VLC: opened=self._open_vlc_by_name()
                    if opened: self.err=""; self._loop(bk)
                except Exception as e:
                    self.err=f"{bk}: {e}"
                finally:
                    with self._lock: self._close_all()
                if self._stop: break
            if not opened:
                self._enqueue(None); time.sleep(0.3)

    # ---------- Open by NAME (preferred), with fallbacks ----------
    def _open_av_by_name(self)->bool:
        opts={}
        if self.s.fps>0: opts["framerate"]=str(self.s.fps)
        if self.s.size!=(0,0): opts["video_size"]=f"{self.s.size[0]}x{self.s.size[1]}"
        pf=AV_PIXELFMT_MAP.get(self.s.fourcc,"")
        if pf: opts["pixel_format"]=pf
        with self._lock:
            self._av_c=av.open(format="dshow", file=f"video={self.s.device_name}", options=opts)
        return True

    def _open_cv2_by_name(self)->bool:
        last_err=None
        dshow_idx = resolve_dshow_index_by_name(self.s.device_name)
        for bkn in self.s.cv2_bks:
            bk = CV2_BK_MAP.get(bkn, None)
            idx = dshow_idx if (bkn=="DSHOW" and dshow_idx is not None) else self.s.index_hint
            cap = cv2.VideoCapture(idx) if (bk is None) else cv2.VideoCapture(idx, bk)
            if not cap.isOpened():
                cap.release(); last_err=f"open_fail[{bkn}]"; continue
            try:
                if self.s.fourcc: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.s.fourcc))
                if self.s.size!=(0,0):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.s.size[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.s.size[1])
                if self.s.fps>0: cap.set(cv2.CAP_PROP_FPS, self.s.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
                with self._lock:
                    self._cv2_cap=cap; self._cv2_bk_used=bkn; self._cv2_idx_used=idx
                return True
            except Exception as e:
                last_err=f"{bkn}:{e}"; cap.release()
        raise RuntimeError(last_err or "cv2_open_fail")

    def _open_vlc_by_name(self)->bool:
        w,h=(self.s.size if self.s.size!=(0,0) else (640,480)); pitch=w*4
        pf = (":dshow-vcodec=mjpg" if self.s.fourcc=="MJPG" else (":dshow-vcodec=uyvy" if self.s.fourcc=="YUY2" else ""))
        size_opt = (f":dshow-size={w}x{h}" if self.s.size!=(0,0) else "")
        mrl=f"dshow://:dshow-vdev={self.s.device_name}{pf}{size_opt}"
        inst=vlc.Instance(VLC_COMMON.split()); media=inst.media_new(mrl)
        player=inst.media_player_new(); player.set_media(media)
        self._vlc_buf=ctypes.create_string_buffer(pitch*h)
        LOCK=ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        UNLOCK=ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        DISP=ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
        self._vlc_lock_cb=LOCK(self._vlc_lock); self._vlc_unlock_cb=UNLOCK(self._vlc_unlock); self._vlc_disp_cb=DISP(self._vlc_display)
        self._opaque=ctypes.py_object(self); opaque=ctypes.cast(ctypes.pointer(self._opaque), ctypes.c_void_p)
        player.video_set_format("RV32", w, h, pitch)
        player.video_set_callbacks(self._vlc_lock_cb, self._vlc_unlock_cb, self._vlc_disp_cb, opaque)
        with self._lock: self._vlc_inst=inst; self._vlc_media=media; self._vlc_player=player
        self._vlc_cb_set=True; player.play()
        return True

    # ---------- Streaming loops ----------
    def _loop(self, bk:str):
        if bk=="AV":
            s=self._av_c.streams.video[0]; s.thread_type="AUTO"
            for pkt in self._av_c.demux(s):
                if self._stop: break
                for fr in pkt.decode():
                    if self._stop: break
                    self._enqueue(fr.to_ndarray(format="bgr24"))
        elif bk=="CV2":
            t0=time.time()
            while not self._stop:
                with self._lock: cap=self._cv2_cap
                if cap is None: break
                ret,img=cap.read()
                if not ret:
                    if time.time()-t0>2: raise RuntimeError(f"read_fail[{self._cv2_bk_used}]")
                    continue
                self._enqueue(img)
        elif bk=="VLC":
            while not self._stop: time.sleep(0.01)

    # ---------- VLC callbacks ----------
    @staticmethod
    def _vlc_lock(opaque, planes):
        self=ctypes.cast(opaque, ctypes.POINTER(ctypes.py_object)).contents.value
        p=ctypes.cast(ctypes.byref(self._vlc_buf), ctypes.c_void_p)
        ctypes.cast(planes, ctypes.POINTER(ctypes.c_void_p))[0]=p
        return p.value
    @staticmethod
    def _vlc_unlock(opaque, picture, planes): pass
    @staticmethod
    def _vlc_display(opaque, picture):
        self=ctypes.cast(opaque, ctypes.POINTER(ctypes.py_object)).contents.value
        w,h=(self.s.size if self.s.size!=(0,0) else (640,480))
        frame=np.frombuffer(self._vlc_buf, dtype=np.uint8).reshape((h,w,4))
        self._enqueue(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))

    # ---------- Queue/cache ----------
    def _enqueue(self, img:Optional[np.ndarray]):
        ts=time.time()
        if img is not None:
            img=img.copy()          # ensure no shared buffer
            self._last_img=img; self._last_ts=ts
        try:
            if self.q.full(): self.q.get_nowait()
        except Empty: pass
        self.q.put_nowait((ts, img))

    def latest(self):
        last_ts=self._last_ts; last_img=self._last_img
        try:
            while True:
                ts,img=self.q.get_nowait()
                if img is not None: last_img=img; last_ts=ts
        except Empty: pass
        self._last_img, self._last_ts=last_img,last_ts
        return last_img, last_ts, self._cv2_bk_used, self._cv2_idx_used

# ---------- Orchestrator ----------
class QuadOrchestrator:
    def __init__(self, specs:List[CamSpec]):
        assert len(specs)==4
        self.specs=specs
        self.grabbers=[FrameGrabber(s) for s in specs]
        self.prev=[None,None,None,None]

    def start(self):  [g.start() for g in self.grabbers]
    def stop(self):   [g.stop()  for g in self.grabbers]

    def _grid(self, frames, labels):
        tiles=[]
        for f in frames:
            if f is None: tiles.append(np.zeros((TILE_H,int(TILE_H*4/3),3),np.uint8)); continue
            scale=TILE_H/max(1,f.shape[0]); w=max(1,int(f.shape[1]*scale))
            tiles.append(cv2.resize(f,(w,TILE_H),interpolation=cv2.INTER_AREA))
        w1=max(tiles[0].shape[1], tiles[1].shape[1]); w2=max(tiles[2].shape[1], tiles[3].shape[1])
        def pad(img,w): 
            if img.shape[1]==w: return img
            return np.hstack([img, np.zeros((img.shape[0], w-img.shape[1], 3), np.uint8)])
        r1=np.hstack([pad(tiles[0],w1), pad(tiles[1],w1)])
        r2=np.hstack([pad(tiles[2],w2), pad(tiles[3],w2)])
        roww=max(r1.shape[1], r2.shape[1]); r1=pad(r1,roww); r2=pad(r2,roww)
        canv=np.vstack([r1,r2])
        for i,lab in enumerate(labels):
            y=(24 if i<2 else r1.shape[0]+24); x=(10 if i%2==0 else roww//2 +10)
            cv2.putText(canv, lab, (x,y), FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)
        return canv

    # --- snapshot helper
    def _rand_tag(self, n=8):
        return "".join(random.choice(string.ascii_uppercase) for _ in range(n))
    def _ensure_dir(self, d): os.makedirs(d, exist_ok=True)
    def _snapshot_all(self, frames):
        self._ensure_dir(CAPTURE_DIR)
        # unique base name
        while True:
            tag=self._rand_tag(8)
            paths=[os.path.join(CAPTURE_DIR, f"{tag}_{i}.png") for i in range(4)]
            if not any(os.path.exists(p) for p in paths): break
        saved=0
        for i,f in enumerate(frames):
            if f is None: continue
            cv2.imwrite(os.path.join(CAPTURE_DIR, f"{tag}_{i}.png"), f)
            saved+=1
        print(f"[CAPTURE] {saved}/4 saved as {tag}_*.png in ./{CAPTURE_DIR}")
        return tag, saved

    def run(self):
        self.start()
        dt=1.0/max(1,DISPLAY_FPS); t_next=time.time()
        try:
            while True:
                frames=[]; labels=[]; now=time.time()
                for i,(s,g) in enumerate(zip(self.specs,self.grabbers)):
                    img, ts, used, idx_used = g.latest()
                    age_ms=(now-ts)*1000 if ts>0 else 1e9
                    if img is None or age_ms>STALE_OK_MS: img=self.prev[i]
                    else: self.prev[i]=img
                    idx_info = f":{idx_used}" if (idx_used is not None) else ""
                    lab=f"{s.title} [{s.device_name}] {s.fourcc or 'DEF'} {s.size[0]}x{s.size[1]} {s.fps or 'DEF'} {('/'+used+idx_info) if used else ''}"
                    if g.err: lab+=" ERR:"+g.err
                    if age_ms<1e8: lab+=f"  {int(max(0,age_ms))}ms"
                    frames.append(img); labels.append(lab)
                canv=self._grid(frames,labels)
                cv2.putText(canv, "Space=snapshot  |  R=restart  |  Q=quit", (10, canv.shape[0]-10), FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW,canv)
                k=cv2.waitKey(1)&0xFF
                if k in (ord('q'),27): break
                if k in (ord('r'),):
                    self.stop(); time.sleep(0.1); self.prev=[None]*4; self.start()
                if k in (ord(' '),):     # snapshot all tiles
                    self._snapshot_all(frames)
                t_next+=dt; sleep=max(0.0, t_next-time.time())
                if sleep>0: time.sleep(sleep)
                else: t_next=time.time()
        finally:
            self.stop(); cv2.destroyAllWindows()

if __name__=="__main__":
    print("FFMPEG available (PyAV):", HAVE_AV, " | VLC available:", HAVE_VLC)
    names=list_dshow_names()
    if names: print("DirectShow devices:", names)
    QuadOrchestrator(CAMS).run()
