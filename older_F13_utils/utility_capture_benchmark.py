# Video pipelines benchmark (results are platform-dependent and pip package version-dependent)
# Simply comment portions of the code corresponding to X if you don't want X installed in your environment

import time
import numpy as np
import av
import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(12)

VIDEO_SOURCE = "rtsp://192.168.88.95:554"
INCOMING_FPS = 25
TEST_DURARION = 5
NUM_FRAMES = INCOMING_FPS * TEST_DURARION
SHOW_FRAMES = False

GST_PIPELINE = f"rtspsrc location={VIDEO_SOURCE} latency=0 ! rtph264depay ! h264parse ! d3d11h264dec ! videoconvert ! video/x-raw,format=I420 ! appsink name=appsink0 sync=false drop=true max-buffers=1"
GST_PIPELINE_SIMPLE = f"rtspsrc location={VIDEO_SOURCE} latency=0 ! rtph264depay ! h264parse ! d3d11h264dec ! appsink name=appsink0"

import ctypes
from ctypes import POINTER, c_char_p, c_void_p, c_int, c_uint, c_size_t
GST_STATE_NULL = 1
GST_STATE_PLAYING = 3
GST_MAP_READ = 1
gst = ctypes.WinDLL("C:\\gstreamer\\1.0\\msvc_x86_64\\bin\\gstreamer-1.0-0.dll")
gst_video = ctypes.WinDLL("C:\\gstreamer\\1.0\\msvc_x86_64\\bin\\gstvideo-1.0-0.dll")
gst_app = ctypes.WinDLL("C:\\gstreamer\\1.0\\msvc_x86_64\\bin\\gstapp-1.0-0.dll")
gst.gst_init.argtypes = [POINTER(c_int), POINTER(POINTER(c_char_p))]
gst.gst_parse_launch.argtypes = [c_char_p, POINTER(c_void_p)]
gst.gst_parse_launch.restype = c_void_p
gst.gst_element_set_state.argtypes = [c_void_p, c_int]
gst.gst_element_set_state.restype = c_int
gst_app.gst_app_sink_pull_sample.argtypes = [c_void_p]
gst_app.gst_app_sink_pull_sample.restype = c_void_p
gst.gst_sample_get_buffer.argtypes = [c_void_p]
gst.gst_sample_get_buffer.restype = c_void_p
gst.gst_sample_get_caps.argtypes = [c_void_p]
gst.gst_sample_get_caps.restype = c_void_p
gst.gst_structure_get_int.argtypes = [c_void_p, c_char_p, POINTER(c_int)]
gst.gst_structure_get_int.restype = c_int
gst.gst_caps_get_structure.argtypes = [c_void_p, c_int]
gst.gst_caps_get_structure.restype = c_void_p
class GstMapInfo(ctypes.Structure):
    _fields_ = [
        ("memory", c_void_p),
        ("flags", c_int),
        ("data", POINTER(ctypes.c_uint8)),
        ("size", c_size_t),
    ]
gst.gst_buffer_map.argtypes = [c_void_p, POINTER(GstMapInfo), c_int]
gst.gst_buffer_map.restype = c_int
gst.gst_buffer_unmap.argtypes = [c_void_p, POINTER(GstMapInfo)]

def benchmark_gstreamer_ctypes(pipeline_desc):
    argc = c_int(0)
    argv = POINTER(c_char_p)()
    gst.gst_init(ctypes.byref(argc), ctypes.byref(argv))
    pipeline = gst.gst_parse_launch(pipeline_desc.encode("utf-8"), None)
    if not pipeline: raise RuntimeError("Failed to create GStreamer pipeline.")
    appsink = gst.gst_bin_get_by_name(pipeline, b"appsink0")
    if not appsink: raise RuntimeError("Failed to get appsink from pipeline.")
    gst.gst_element_set_state(pipeline, GST_STATE_PLAYING)
    frame_count = 0
    width, height = None, None
    start_time = time.time()
    try:
        while frame_count < NUM_FRAMES:
            sample = gst_app.gst_app_sink_pull_sample(appsink)
            if not sample: raise RuntimeError("Failed to pull sample from appsink.")
            buffer = gst.gst_sample_get_buffer(sample)
            if not buffer: raise RuntimeError("Failed to get buffer from sample.")
            caps = gst.gst_sample_get_caps(sample)
            if not caps: raise RuntimeError("Failed to get caps from sample.")
            if width is None or height is None:
                structure = gst.gst_caps_get_structure(caps, 0)
                if not structure:
                    raise RuntimeError("Failed to get structure from caps.")
                w, h = c_int(), c_int()
                gst.gst_structure_get_int(structure, b"width", ctypes.byref(w))
                gst.gst_structure_get_int(structure, b"height", ctypes.byref(h))
                width, height = w.value, h.value
                print(f"Stream resolution: {width}x{height}")
            map_info = GstMapInfo()
            success = gst.gst_buffer_map(buffer, ctypes.byref(map_info), GST_MAP_READ)
            if not success: raise RuntimeError("Failed to map buffer.")
            frame_size = width * height * 3 // 2
            if map_info.size != frame_size:
                gst.gst_buffer_unmap(buffer, ctypes.byref(map_info))
                raise ValueError(f"Buffer size mismatch: expected {frame_size}, got {map_info.size}")
            yuv_frame = np.ctypeslib.as_array(map_info.data, shape=(map_info.size,))
            gst.gst_buffer_unmap(buffer, ctypes.byref(map_info))
            if SHOW_FRAMES:
                y = yuv_frame[: width * height].reshape((height, width))
                cv2.imshow("Frame", y)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            frame_count += 1
    finally:
        gst.gst_element_set_state(pipeline, GST_STATE_NULL)
        elapsed_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds.")
    return frame_count, elapsed_time

def print_stream_info_opencv(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened(): raise ValueError("Failed to open video source with OpenCV.")
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"OpenCV Stream Information: Codec: {codec:08X}, Resolution: {width}x{height}, FPS: {fps}")
    cap.release()

def print_stream_info_pyav(video_source):
    container = av.open(video_source)
    print("PyAV Stream Information:")
    for stream in container.streams: print(f"Stream {stream.index}: {stream.type}, Codec: {stream.codec_context.name}, Bitrate: {stream.bit_rate}, FPS: {stream.average_rate}")

def benchmark_opencv(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened(): raise ValueError("Failed to open video source with OpenCV.")
    start_time = time.time()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (NUM_FRAMES and frame_count >= NUM_FRAMES): break
        frame_count += 1
        if SHOW_FRAMES:
            cv2.imshow("OpenCV", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    elapsed_time = time.time() - start_time
    cap.release()
    if SHOW_FRAMES: cv2.destroyAllWindows()
    print_stream_info_opencv(video_source)
    return frame_count, elapsed_time

def benchmark_opencv_gstreamer(video_source):
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened(): raise ValueError("Failed to open video source with OpenCV GStreamer.")
    start_time = time.time()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (NUM_FRAMES and frame_count >= NUM_FRAMES): break
        frame_count += 1
        if SHOW_FRAMES:          
            cv2.imshow("OpenCV GStreamer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    elapsed_time = time.time() - start_time
    cap.release()
    if SHOW_FRAMES: cv2.destroyAllWindows()
    print_stream_info_opencv(GST_PIPELINE)
    return frame_count, elapsed_time

def benchmark_pyav(video_source):
    container = av.open(video_source)
    video_streams = [s for s in container.streams if s.type == "video"]
    if not video_streams: raise ValueError("No video streams found in the container.")
    video_stream = video_streams[0]  # Use the first video stream
    print(f"PyAV Stream Information:")
    print(f"Stream {video_stream.index}: {video_stream.type}, Codec: {video_stream.codec_context.name}, Bitrate: {video_stream.bit_rate}, FPS: {video_stream.average_rate or 'unknown'}")
    start_time = time.time()
    frame_count = 0
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            frame_count += 1
            if NUM_FRAMES and frame_count >= NUM_FRAMES: break
            if SHOW_FRAMES:
                frame_img = frame.to_ndarray(format="bgr24")
                cv2.imshow("PyAV", frame_img)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
        if NUM_FRAMES and frame_count >= NUM_FRAMES: break
    elapsed_time = time.time() - start_time
    if SHOW_FRAMES: cv2.destroyAllWindows()
    return frame_count, elapsed_time

def benchmark_gstreamer_direct(video_source):
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    pipeline = Gst.parse_launch(GST_PIPELINE)
    appsink = pipeline.get_by_name("appsink0")
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False)    
    if not appsink: raise ValueError("Failed to create GStreamer appsink.")
    pipeline.set_state(Gst.State.PLAYING)
    start_time = time.time()
    frame_count = 0
    width, height = None, None
    while frame_count < NUM_FRAMES:
        sample = appsink.emit("pull-sample")
        if not sample: break
        buf = sample.get_buffer()
        caps = sample.get_caps()        
        if width is None or height is None:
            width = caps.get_structure(0).get_value("width")
            height = caps.get_structure(0).get_value("height")
        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            data = map_info.data
            frame_size = len(data)
            expected_frame_size = width * height * 3 // 2 
            if frame_size != expected_frame_size: raise ValueError(f"Buffer size mismatch: expected {expected_frame_size}, got {frame_size}")
            yuv_frame = np.frombuffer(data, dtype=np.uint8)
            if SHOW_FRAMES:
                y = yuv_frame[: width * height].reshape((height, width))
                u = yuv_frame[width * height : width * height + (width // 2) * (height // 2)].reshape((height // 2, width // 2))
                v = yuv_frame[width * height + (width // 2) * (height // 2) :].reshape((height // 2, width // 2))
                u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
                v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
                bgr_frame = cv2.merge((y, u, v))
                cv2.imshow("GStreamer Direct", bgr_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            buf.unmap(map_info)
            frame_count += 1
    elapsed_time = time.time() - start_time
    pipeline.set_state(Gst.State.NULL)
    if SHOW_FRAMES: cv2.destroyAllWindows()
    print(f"GStreamer Stream Information: Resolution: {width}x{height}")
    return frame_count, elapsed_time

def main():
    print("Starting benchmarks...\n")
    try:
        print("\nBenchmarking GStreamer Direct:")
        frames, time_taken = benchmark_gstreamer_direct(VIDEO_SOURCE)
        print(f"GStreamer Direct: Processed {frames} frames in {time_taken:.2f} seconds ({frames / time_taken:.2f} FPS).")
    except Exception as e: print(f"GStreamer Direct benchmark failed: {e}")
    try:
        print("\nBenchmarking GStreamer Direct with ctypes:")
        frames, time_taken = benchmark_gstreamer_ctypes(GST_PIPELINE)
        print(f"GStreamer Direct (ctypes): Processed {frames} frames in {time_taken:.2f} seconds ({frames / time_taken:.2f} FPS).")
    except Exception as e: print(f"GStreamer Direct (ctypes) benchmark failed: {e}")
    try:
        print("Benchmarking OpenCV:")
        frames, time_taken = benchmark_opencv(VIDEO_SOURCE)
        print(f"OpenCV: Processed {frames} frames in {time_taken:.2f} seconds ({frames / time_taken:.2f} FPS).")
    except Exception as e: print(f"OpenCV benchmark failed: {e}")
    try:
        print("\nBenchmarking PyAV:")
        frames, time_taken = benchmark_pyav(VIDEO_SOURCE)
        print(f"PyAV: Processed {frames} frames in {time_taken:.2f} seconds ({frames / time_taken:.2f} FPS).")
    except Exception as e: print(f"PyAV benchmark failed: {e}")
    try:
        print("\nBenchmarking OpenCV with GStreamer:")
        frames, time_taken = benchmark_opencv_gstreamer(VIDEO_SOURCE)
        print(f"OpenCV GStreamer: Processed {frames} frames in {time_taken:.2f} seconds ({frames / time_taken:.2f} FPS).")
    except Exception as e: print(f"OpenCV GStreamer benchmark failed: {e}")

if __name__ == "__main__": main()