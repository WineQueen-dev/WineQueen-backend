from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from ultralytics import YOLO
import cv2
import threading
import json
import time
import os
import asyncio
from queue import Queue
import serial  # 시리얼 통신 라이브러리
import numpy as np # 처음에 바로 yolo키기 위한 임포트

# =========================
# 전역 설정/상수
# ========================
model = YOLO("best_wCrop.pt")
model.fuse()

# 카메라/시리얼 전역 핸들 (❗️추가: cap 초기화)
cap = None
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
ser = None

# YOLO 토글
YOLO_ON = False
def set_yolo_active(active: bool):
    """YOLO 추론 사용 토글 (카메라는 그대로 유지)."""
    global YOLO_ON
    with state_lock:
        YOLO_ON = active
    print(f"[YOLO] {'ON' if active else 'OFF'}")

# 버튼 이벤트 큐
last_button = None
button_queue = Queue()

# 상태 머신
STAY = "STAY"
ALIGNING = "ALIGNING"
SEALING = "SEALING"
OPENING = "OPENING"

SYSTEM_STATE = STAY
TARGET_ACTION = None  # 'S' or 'O'

state_lock = threading.Lock()
serial_lock = threading.Lock()

# 클라이언트 및 프레임 공유
clients = []
latest_annotated_frame = None
latest_detections_json = "{}"
frame_lock = threading.Lock()
detections_lock = threading.Lock()

# =========================
# 하드웨어 보조 함수
# =========================
def ensure_serial_open():
    global ser
    if ser is None or not ser.is_open:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"[SER] open OK {SERIAL_PORT} {BAUD_RATE}")
            time.sleep(2)  # 아두이노 리셋 대기
            return True
        except Exception as e:
            print(f"[SER] open FAIL: {e}")
            ser = None
            return False
    return True

def ensure_camera_open():
    global cap
    if cap is not None and cap.isOpened():
        return True
    
    try:
        device = "/dev/video0"
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        
        # ✅ 추가 최적화 설정
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)  # ✅ 30에서 20으로
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # ✅ MJPEG 코덱 사용
        
        if cap.isOpened():
            # ✅ 초기 프레임 몇 개 버리기 (카메라 안정화)
            for _ in range(5):
                cap.read()
            print(f"[CAM] open OK {device}")
            return True
        else:
            print(f"[CAM] open FAIL {device}")
            cap = None
            return False
    except Exception as e:
        print(f"[CAM] exception: {e}")
        cap = None
        return False

# =========================
# 감지 루프(스레드)
# =========================

def detection_loop():
    global latest_annotated_frame, latest_detections_json, cap, ser, SYSTEM_STATE, TARGET_ACTION
    global YOLO_ON
    
    last_serial_send_time = 0
    last_alignment_check = 0  # ✅ 정렬 체크 타이밍 추가
    frame_skip_counter = 0
    TARGET_FPS = 20  # ✅ 30에서 20으로 감소 (부하 감소)
    frame_interval = 1.0 / TARGET_FPS
    
    while True:
        loop_start_time = time.time()
        
        if not ensure_camera_open():
            time.sleep(1.0)
            continue

        # ✅ 버퍼 비우기 (최신 프레임만 가져오기)
        for _ in range(2):  # 버퍼에 있는 오래된 프레임 제거
            cap.grab()
        
        ret, frame = cap.read()
        if not ret:
            print("[CAM] read fail, will retry...")
            time.sleep(0.1)
            continue
        
        with state_lock:
            current_state = SYSTEM_STATE
            yolo_active = YOLO_ON
            target_action = TARGET_ACTION
        
        # 카메라 중심 표시 코드는 그대로...
        h, w, _ = frame.shape
        cam_center_x, cam_center_y = w // 2, h // 2
        cv2.line(frame, (0, cam_center_y), (w, cam_center_y), (0, 255, 255), 1)
        cv2.circle(frame, (cam_center_x, cam_center_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"(X:{cam_center_x}, Y:{cam_center_y})", (cam_center_x + 10, cam_center_y + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        detections = []
        results = None
        
        if yolo_active:
            frame_skip_counter += 1
            should_run_yolo = (frame_skip_counter % 3 == 0)  # ✅ 3프레임당 1번으로 변경
            
            if should_run_yolo:
                # ✅ YOLO 추론 최적화
                results = model(frame, 
                              classes=0, 
                              conf=0.7, 
                              verbose=False,
                              imgsz=320,  # 고정 크기 -> 잘 안되면 416으로 변경하기.
                              device='cpu',  # CPU 명시
                              max_det=1)[0]  # 최대 1개 객체만 검출
            
            now = time.time()
            if current_state == ALIGNING and results is not None:
                # 제어 신호 전송 최적화
                if now - last_serial_send_time >= 0.3:  # 일정한 간격
                    if results.boxes and len(results.boxes) > 0:
                        first_box = results.boxes[0]
                        x1, y1, x2, y2 = map(int, first_box.xyxy[0])
                        obj_center_y = (y1 + y2) // 2
                        relative_x = obj_center_y - cam_center_y
                        
                        print(f"[ALIGN] Camera center: {cam_center_y}, Wine center: {obj_center_y}, Relative: {relative_x}")
                        
                        deadzone_pixels = 5
                        
                        if abs(relative_x) <= deadzone_pixels:
                            # 정렬 완료 확인 로직 개선
                            if last_alignment_check == 0:
                                last_alignment_check = now  # 0.5초 동안 정렬 유지 확인
                                
                            elif now - last_alignment_check >= 0.5:
                                send_serial_command('C\n', show_log=True)
                                print("[ALIGN] Alignment confirmed! Sending target action...")
                                
                                time.sleep(0.2)  # 0.1에서 0.2로 증가
                                if target_action:
                                    send_serial_command(target_action, show_log=True)
                                    with state_lock:
                                        SYSTEM_STATE = SEALING if target_action == 'S\n' else OPENING
                                else:
                                    print("[ERROR] No target action set!")
                                    with state_lock:
                                        SYSTEM_STATE = STAY
                                
                                set_yolo_active(False)
                                last_alignment_check = 0
                        else:
                            last_alignment_check = 0  # 정렬 벗어나면 리셋
                            
                            if relative_x < -deadzone_pixels:
                                send_serial_command('R\n', show_log=True)
                            else:
                                send_serial_command('L\n', show_log=True)
                        
                        last_serial_send_time = now
                    else:
                        print("[ALIGN] No wine bottle detected")
                        # ✅ 와인병 미감지 시 정지 신호
                        if now - last_serial_send_time >= 1.0:  # 1초마다
                            last_serial_send_time = now

            # 화면 그리기 코드는 그대로...
            if results and results.boxes and len(results.boxes) > 0:
                first_box = results.boxes[0]
                x1, y1, x2, y2 = map(int, first_box.xyxy[0])
                conf = float(first_box.conf[0])
                class_id = int(first_box.cls[0])
                class_name = model.names[class_id]
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2
                relative_x = obj_center_x - cam_center_x
                relative_y = cam_center_y - obj_center_y

                detections.append({
                    "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                    "conf": conf, "class_id": class_id, "class_name": class_name,
                    "relative_center": {"x": relative_x, "y": relative_y}
                })
                
                # 박스 및 텍스트 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                coord_text = f"({relative_x}, {relative_y})"
                cv2.putText(frame, coord_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        else:
            cv2.putText(frame, "YOLO: OFF", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 상태 표시
        status_text = f"State: {current_state}"
        if target_action:
            # 1. 어떤 작업인지(SEAL 또는 OPEN)를 먼저 결정해서 변수에 담습니다.
            target_display = "SEAL" if target_action == 'S\n' else "OPEN"
            # 2. 그 변수를 f-string 안에서 사용합니다.
            status_text += f" | Target: {target_display}"

        cv2.putText(frame, status_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 결과 저장 (기존 코드와 동일)
        with detections_lock:
            latest_detections_json = json.dumps({
                "timestamp": time.time(), 
                "detections": detections,
                "state": current_state,
                "target": target_action  # ✅ 타겟 액션 추가
            })

        with frame_lock:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # ✅ 85에서 80으로
            latest_annotated_frame = buffer.tobytes()

        # FPS 제어
        elapsed = time.time() - loop_start_time
        sleep_time = max(0, frame_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

# =========================
# WS 브로드캐스터
# =========================
async def broadcast_detections():
    while True:
        with detections_lock:
            data = latest_detections_json
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                clients.remove(ws)
            except:
                pass
        await asyncio.sleep(0.1)

# =========================
# FastAPI 앱/라이프사이클
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # YOLO Warm-up 코드 추가
    print("YOLO model warming up...")
    # 더미(dummy) 이미지 생성 (실제 이미지와 크기, 채널을 맞추는 것이 좋음)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8) 
    # 모델을 한 번 실행하여 미리 준비시킴
    model(dummy_image, verbose=False) 
    print("YOLO model is ready.")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    print("서버 시작: 감지 스레드 및 브로드캐스터 시작...")
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()

    serial_thread = threading.Thread(target=serial_reader_loop, daemon=True)
    serial_thread.start()

    broadcast_task = asyncio.create_task(broadcast_detections())
    broadcast_btn_task = asyncio.create_task(broadcast_buttons())
    try:
        yield
    finally:
        print("서버 종료...")
        broadcast_task.cancel()
        broadcast_btn_task.cancel()
        if cap and cap.isOpened():
            cap.release()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =========================
# 시리얼 전송/컨트롤 API
# =========================
def send_serial_command(command: str, show_log: bool = True):
    if not ensure_serial_open():
        if show_log: print("Serial port not open")
        return False, "Serial port not open"
    try:
        with serial_lock:
            ser.write(command.encode('utf-8'))
            ser.flush()
        if show_log: print(f"Serial command sent: '{command.strip()}'")
        return True, f"Command '{command.strip()}' sent"
    except Exception as e:
        if show_log: print(f"Serial command send fail: {e}")
        return False, str(e)

@app.post("/control/seal", tags=["Arduino Control"])
async def start_sealing():
    """밀봉을 위한 정렬 프로세스를 시작합니다."""
    global SYSTEM_STATE, TARGET_ACTION
    with state_lock:
        if SYSTEM_STATE != STAY:
            return JSONResponse(status_code=409, content={"status": "error", "message": f"System is busy with '{SYSTEM_STATE}'"})
        SYSTEM_STATE = ALIGNING
        TARGET_ACTION = 'S\n'
    print(f"SystemState Change: {STAY} -> {ALIGNING} (목표: 밀봉)")
    return {"status": "ok", "message": "Alignment process for sealing has been started."}

@app.post("/control/open", tags=["Arduino Control"])
async def start_opening():
    """개봉을 위한 정렬 프로세스를 시작합니다."""
    global SYSTEM_STATE, TARGET_ACTION
    with state_lock:
        if SYSTEM_STATE != STAY:
            return JSONResponse(status_code=409, content={"status": "error", "message": f"System is busy with '{SYSTEM_STATE}'"})
        SYSTEM_STATE = ALIGNING
        TARGET_ACTION = 'O\n'
    print(f"SystemState Change: {STAY} -> {ALIGNING} (목표: 개봉)")
    return {"status": "ok", "message": "Alignment process for opening has been started."}

@app.post("/control/home", tags=["Arduino Control"])
async def return_to_home():
    success, message = send_serial_command('H\n')
    if success:
        return {"status": "ok", "message": message}
    return JSONResponse(status_code=500, content={"status": "error", "message": message})

@app.post("/control/stop", tags=["Arduino Control"])
async def emergency_stop():
    global SYSTEM_STATE, TARGET_ACTION
    with state_lock:
        SYSTEM_STATE = STAY
        TARGET_ACTION = None
    set_yolo_active(False)
    success, message = send_serial_command('E\n')
    if success:
        return {"status": "ok", "message": f"{message}. System state has been reset to '{STAY}'."}
    return JSONResponse(status_code=500, content={"status": "error", "message": message})

# =========================
# WebSocket
# =========================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    print(f"[PID {os.getpid()}] WS connect. clients={len(clients)}")
    await websocket.send_text(json.dumps({"type": "connected", "ts": time.time()}))

    async def sender():
        while True:
            with detections_lock:
                data = latest_detections_json
            await websocket.send_text(data)
            await asyncio.sleep(0.1)

    async def receiver():
        while True:
            msg = await websocket.receive_text()
            t = (msg or "").strip()
            if t == "ping" or t == '{"type":"ping"}':
                await websocket.send_text(json.dumps({"type":"pong","ts":time.time()}))
                continue

    try:
        await asyncio.gather(sender(), receiver())
    except WebSocketDisconnect:
        pass
    finally:
        try:
            clients.remove(websocket)
        except Exception:
            pass
        print(f"[PID {os.getpid()}] WS disconnect. clients={len(clients)}")

# =========================
# MJPEG 스트림
# =========================
def generate_annotated_frame():
    while True:
        with frame_lock:
            if latest_annotated_frame is None:
                time.sleep(0.03)
                continue
            frame_bytes = latest_annotated_frame
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.02)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_annotated_frame(), media_type="multipart/x-mixed-replace; boundary=frame")

# =========================
# 정적 파일(React)
# =========================
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="static")
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str = ""):
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"error": "Frontend not found"}
else:
    @app.get("/")
    def root():
        return {"message": "Backend is running, but frontend build is not found."}

# =========================
# 시리얼 수신(스레드)
# =========================
def serial_reader_loop():
    global ser, SYSTEM_STATE, TARGET_ACTION
    while True:
        if not ensure_serial_open():
            time.sleep(0.5)
            continue
        try:
            with serial_lock:
                raw = ser.readline()
                if not raw:
                    continue
            
            line = raw.decode(errors="ignore").strip()
            print(f"[SERIAL READ] Raw data: '{line}'")

            # 아두이노 명령 처리
            if line == 'A':
                print("[SERIAL READ] 'A' - Arduino requesting alignment")
                set_yolo_active(True)
                with state_lock:
                    SYSTEM_STATE = ALIGNING
                    # TARGET_ACTION은 이미 버튼으로 설정되어 있어야 함
                    
            elif line == 'F':
                print("[SERIAL READ] 'F' - Process finished")
                set_yolo_active(False)
                with state_lock:
                    SYSTEM_STATE = STAY
                    TARGET_ACTION = None
            
            # 버튼 이벤트 처리
            elif line == '1':
                print("[SERIAL READ] Button 1 - Seal process")
                with state_lock:
                    TARGET_ACTION = 'S\n'  # ✅ TARGET_ACTION 설정 추가
                button_queue.put_nowait("SEAL_REDIRECT")
                
            elif line == '2':
                print("[SERIAL READ] Button 2 - Open process")
                with state_lock:
                    TARGET_ACTION = 'O\n'  # ✅ TARGET_ACTION 설정 추가
                button_queue.put_nowait("OPEN_REDIRECT")

        except Exception as e:
            print(f"시리얼 읽기 오류: {e}")
            try:
                with serial_lock:
                    if ser:
                        ser.close()
            except:
                pass
            ser = None
            time.sleep(1)

# =========================
# 버튼 이벤트 WS 브로드캐스트
# =========================
async def broadcast_buttons():
    loop = asyncio.get_running_loop()
    while True:
        event = await loop.run_in_executor(None, button_queue.get)
        if event == "SEAL_REDIRECT":
            payload = json.dumps({"type": "redirect", "page": "/seal"})
            print(f"[WS] redirect /seal → {len(clients)} clients")
        elif event == "OPEN_REDIRECT":
            payload = json.dumps({"type": "redirect", "page": "/open"})
            print(f"[WS] redirect /open → {len(clients)} clients")
        else:
            payload = json.dumps({"type": "button", "value": event, "ts": time.time()})
            print(f"[WS] broadcasting button={event} → {len(clients)} clients")

        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                clients.remove(ws)
            except:
                pass

# =========================
# 엔트리포인트
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
