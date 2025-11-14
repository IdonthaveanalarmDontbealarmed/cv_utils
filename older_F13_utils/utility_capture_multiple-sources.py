import cv2, numpy as np, os, random, string, time

SELFIE=True

ZOOM_RANGE,HUE_RANGE,QUALITY_RANGE=(2000,16000),(0,7),(0,2)
PROP_NAMES=[None]*100
PROP_NAMES[3]="Cap Width"
PROP_NAMES[4]="Cap Height"
PROP_NAMES[10]="Brightness"
PROP_NAMES[11]="Contrast"
PROP_NAMES[12]="[1-3] Quality"
PROP_NAMES[13]="[A-D] Tint"
PROP_NAMES[27]="[W-S] Zoom"
PROP_NAMES[37]="[E] UVC GUI"

def rand_id(n=8): 
    return "".join(random.choice(string.ascii_uppercase) for _ in range(n))

def list_uvc_props(cap):
    p={}
    for i in range(100):
        v=cap.get(i)
        if v!=-1: p[i]=v
    return p

def set_init_uvc(cap):
    cap.set(12,QUALITY_RANGE[-1])
    cap.set(27,ZOOM_RANGE[0])

def toggle_zoom(cap,v):
    z=int(cap.get(27))
    cap.set(27,np.clip(z+v,*ZOOM_RANGE))

def adjust_tint(cap,v):
    t=int(cap.get(13))
    cap.set(13,np.clip(t+v,*HUE_RANGE))

def set_quality(cap,l):
    if l in [0,1,2]: cap.set(12,l)

def print_keys():
    print("\n--- Keyboard Controls ---")
    print("W/S - Zoom (UVC)")
    print("A/D - Tint (UVC)")
    print("1/2/3 - Quality (UVC)")
    print("E - Open UVC GUI")
    print("Space - Save frames")
    print("Q - Quit")
    print("-------------------------")

def main():
    if not os.path.exists("CAP"): os.makedirs("CAP")
    cap_t=cv2.VideoCapture(1,cv2.CAP_DSHOW)
    if not cap_t.isOpened():
        print("Error: UVC camera [1] not opened.")
        return
    set_init_uvc(cap_t)
    props=list_uvc_props(cap_t)
    cap_r=cv2.VideoCapture("rtsp://192.168.88.95:554/av0_0")
    cap_s=cv2.VideoCapture("rtsp://192.168.88.96:554/av0_0")
    if not (cap_r.isOpened() and cap_s.isOpened()):
        print("Error: RTSP stream(s) not opened.")
        return
    cap_selfie=None
    if SELFIE:
        cap_selfie=cv2.VideoCapture(0)
        if not cap_selfie.isOpened():
            print("Warning: Selfie camera [0] not opened.")
            cap_selfie=None
    print_keys()
    blink=0; fps=0.0; pt=None
    while True:
        t0=time.time()
        ret_t,frm_t=cap_t.read()
        ret_r,frm_r=cap_r.read()
        ret_s,frm_s=cap_s.read()
        if not (ret_t and ret_r and ret_s): break
        ft_rz=cv2.resize(frm_t,(640,480))
        fr_rz=cv2.resize(frm_r,(640,480))
        fs_rz=cv2.resize(frm_s,(640,480))
        disp=np.zeros((960,1280,3),dtype=np.uint8)
        disp[0:480,0:640]=ft_rz
        disp[480:960,0:640]=fr_rz
        disp[480:960,640:1280]=fs_rz
        if cap_selfie:
            ret_sf,frm_sf=cap_selfie.read()
            if ret_sf:
                sf_rz=cv2.resize(frm_sf,(640,480))
                disp[0:480,640:1280]=sf_rz
        for k in props.keys():
            props[k]=cap_t.get(k)
        y=30
        cv2.putText(disp,f"FPS:{fps:.1f}",(650,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        y+=30
        for k,v in props.items():
            nm=PROP_NAMES[k] if PROP_NAMES[k] else f"Prop {k}"
            cv2.putText(disp,f"{nm}:{v:.1f}",(650,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            y+=25
        if blink>0:
            cv2.rectangle(disp,(1050,30),(1100,80),(0,0,255),-1)
            blink-=1
        cv2.imshow("Multi-Stream",disp)
        k=cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        elif k==ord('w'): toggle_zoom(cap_t,1000)
        elif k==ord('s'): toggle_zoom(cap_t,-1000)
        elif k==ord('a'): adjust_tint(cap_t,-1)
        elif k==ord('d'): adjust_tint(cap_t,1)
        elif k==ord('e'): cap_t.set(37,1)
        elif k==ord('1'): set_quality(cap_t,0)
        elif k==ord('2'): set_quality(cap_t,1)
        elif k==ord('3'): set_quality(cap_t,2)
        elif k==32:
            uid=rand_id()
            cv2.imwrite(f"CAP/cap_{uid}_t.jpg",frm_t)
            cv2.imwrite(f"CAP/cap_{uid}_r.jpg",frm_r)
            cv2.imwrite(f"CAP/cap_{uid}_s.jpg",frm_s)
            blink=15
        if pt!=None:
            dt=t0-pt
            if dt>0: fps=0.9*fps+0.1*(1./dt)
        pt=t0
    cap_t.release()
    cap_r.release()
    cap_s.release()
    if cap_selfie: cap_selfie.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
