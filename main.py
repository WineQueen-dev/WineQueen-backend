from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from ultralytics import YOLO
import cv2
import threading
import json
import time
import os
import asyncio

import serial # 시리얼 통신 라이브러리 

# --- 전역 변수 및 설정 ---
# YOLO 모델 로드 (학습된 커스텀 모델)
model = YOLO("best_wCrop.pt") 

# 카메라 설정
try:
    # [변경점] 카메라 인덱스(8) 대신, 터미널에서 확인한 장치 경로를 직접 입력합니다.
    # 예시로 /dev/video0 을 사용했으며, 실제 확인된 경로로 수정해주세요.
    CAMERA_DEVICE_PATH = "/dev/video0" 
    cap = cv2.VideoCapture(CAMERA_DEVICE_PATH)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam: {CAMERA_DEVICE_PATH}")
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"카메라 초기화 성공: {CAMERA_DEVICE_PATH}")

except Exception as e:
    print(f"카메라 초기화 실패: {e}")
    cap = None

# 시리얼 통신 설정
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 9600 # 아두이노 스케치에서 설정한 보드레이트와 동일하게 맞춰야 합니다.
ser = None # 시리얼 객체 초기화

# --- [추가] 상태 머신을 위한 전역 변수 ---
STAY = "STAY"
ALIGNING = "ALIGNING"
SEALING = "SEALING"
OPENING = "OPENING"

SYSTEM_STATE = STAY
TARGET_ACTION = None # 정렬 후 수행할 작업 ('S' 또는 'O')

state_lock = threading.Lock()
serial_lock = threading.Lock()
# --- [추가] 끝 ---


# 클라이언트 및 데이터 공유를 위한 변수
clients = []
latest_annotated_frame = None
latest_detections_json = "{}"
frame_lock = threading.Lock()
detections_lock = threading.Lock()

# --- 객체 감지 및 데이터 처리 (백그라운드 스레드 - 생산자) ---
def detection_loop():
    global latest_annotated_frame, latest_detections_json, cap, ser, SYSTEM_STATE, TARGET_ACTION

    if cap is None:
        print("카메라가 없으므로 감지 루프를 시작할 수 없습니다.")
        return

    # 시리얼 포트 열기
    while ser is None:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"시리얼 포트 {SERIAL_PORT} 열기 성공")
            time.sleep(2) # 아두이노 보드 초기화 시간 대기
        except Exception as e:
            print(f"시리얼 포트 열기 실패: {e}")
            time.sleep(2) # 2초후 재시도
            ser = None


    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패. 루프를 종료합니다.")
            break
        
         # --- [수정] 비디오 스트리밍 멈춤 없는 상태 관리 ---
        with state_lock:
            current_state = SYSTEM_STATE
            
            # 밀봉 또는 개봉 작업이 진행 중일 때, 시간이 다 되었는지 확인
            if current_state in [SEALING, OPENING] and PROCESS_START_TIME is not None:
                if time.time() - PROCESS_START_TIME > PROCESS_DURATION:
                    print(f"작업 시간({PROCESS_DURATION}초) 경과. 시스템 상태를 '{STAY}'로 복귀합니다.")
                    SYSTEM_STATE = STAY
                    TARGET_ACTION = None
                    PROCESS_START_TIME = None
        # --- [수정] 끝 ---

        # --- 카메라 중심 좌표 계산 및 표시 ---
        h, w, _ = frame.shape
        cam_center_x, cam_center_y = w // 2, h // 2

        # 카메라 중심점에 파란색 원과 (0, 0) 텍스트 표시
        cv2.circle(frame, (cam_center_x, cam_center_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, "(0, 0)", (cam_center_x + 10, cam_center_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # --- 끝 ---

        # 1. AI 추론 (여기서 딱 한 번만 실행)
        results = model(frame, classes=0, conf=0.9, verbose=False)[0]
        
        detections = []
        # 2. 결과 처리 및 화면 그리기
        # --- 상태가 ALIGNING일 때만 위치 보정 신호를 보냄 
        if current_state == ALIGNING:
            serial_data_to_send = '2' # 기본값: 동일 (2)
        
            if results.boxes:
                first_box = results.boxes[0]
                x1, y1, x2, y2 = map(int, first_box.xyxy[0])
                # --- 객체 중심의 상대 좌표 계산 ---
                obj_center_x = (x1 + x2) // 2
                # 카메라 중심을 (0,0)으로 하는 상대 좌표
                relative_x = obj_center_x - cam_center_x

                # --- 시리얼 통신을 위한 데이터 결정 (데드존 적용) ---
                deadzone_pixels = 10  # 좌우 10픽셀을 데드존으로 설정
                
                if relative_x < -deadzone_pixels:
                    serial_data_to_send = '0' # 와인이 중심보다 왼쪽에 있음
                elif relative_x > deadzone_pixels:
                    serial_data_to_send = '1' # 와인이 중심보다 오른쪽에 있음
                else:
                    serial_data_to_send = '2' # 와인이 중앙 데드존 안에 위치함 (정렬 완료)
                    # 정렬 완료 시, 목표했던 작업(밀봉/개봉) 신호 전송
                    if TARGET_ACTION:
                        print(f"정렬 완료! 목표 작업 '{TARGET_ACTION}' 신호를 전송합니다.")
                        send_serial_command(TARGET_ACTION)
                        
                        # 상태 변경
                        if TARGET_ACTION == 'S':
                            SYSTEM_STATE = SEALING
                        elif TARGET_ACTION == 'O':
                            SYSTEM_STATE = OPENING
                        
                        PROCESS_START_TIME = time.time() # 작업 시작 시간 기록
                        TARGET_ACTION = None # 목표 작업 초기화
                
            else: # 감지된 객체가 없을 경우(예외처리)
                serial_data_to_send = '3' # 예를 들어, 와인이 감지되지 않았음을 알리는 코드 (선택 사항)
                                        # 아두이노에서 이 경우를 어떻게 처리할지 정의해야 함
            
            # 3. 정렬 신호 전송
            send_serial_command(serial_data_to_send, show_log=False)
        
        # 웹소켓 및 영상 스트리밍을 위한 화면 그리기는 상태와 상관없이 항상 수행
        if results.boxes:
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            coord_text = f"({relative_x}, {relative_y})"
            cv2.putText(frame, coord_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 4. 처리된 결과물을 스레드 안전하게 전역 변수에 저장
        with detections_lock:
            latest_detections_json = json.dumps({"timestamp": time.time(), "detections": detections})

        with frame_lock:
            _, buffer = cv2.imencode('.jpg', frame)
            latest_annotated_frame = buffer.tobytes()

        time.sleep(0.03) # CPU 사용량 조절

    if ser and ser.is_open:
        ser.close() # 루프 종료 시 시리얼 포트 닫기
    cap.release()

# --- 웹소켓 데이터 브로드캐스팅 (소비자) ---
async def broadcast_detections():
    while True:
        try:
            with detections_lock:
                detections_to_send = latest_detections_json
            
            # 비동기적으로 모든 클라이언트에게 메시지 전송
            await asyncio.gather(
                *[ws.send_text(detections_to_send) for ws in clients],
                return_exceptions=False
            )
        except Exception as e:
            print(f"브로드캐스팅 오류: {e}")

        await asyncio.sleep(0.1) # 0.1초마다 데이터 전송

# --- FastAPI Lifespan 및 라우트 설정 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("서버 시작: 감지 스레드 및 브로드캐스터 시작...")
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    
    broadcast_task = asyncio.create_task(broadcast_detections())
    
    yield
    
    print("서버 종료...")
    broadcast_task.cancel()
    if cap and cap.isOpened():
        cap.release()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

## ----------- 아두이노 제어 API (상태 머신 적용) ---------------- ##

def send_serial_command(command: str, show_log: bool = True):
    if not ser or not ser.is_open:
        if show_log: print("시리얼 포트가 열려있지 않습니다.")
        return False, "Serial port not open"
    try:
        with serial_lock:
            ser.write(command.encode('utf-8'))
        if show_log: print(f"Serial command sent: '{command}'")
        return True, f"Command '{command}' sent"
    except Exception as e:
        if show_log: print(f"시리얼 명령 전송 실패: {e}")
        return False, str(e)

@app.post("/control/seal", tags=["Arduino Control"])
async def start_sealing():
    """밀봉을 위한 정렬 프로세스를 시작합니다."""
    global SYSTEM_STATE, TARGET_ACTION
    with state_lock:
        if SYSTEM_STATE != STAY:
            return JSONResponse(status_code=409, content={"status": "error", "message": f"System is busy with '{SYSTEM_STATE}'"})
        SYSTEM_STATE = ALIGNING
        TARGET_ACTION = 'S'
        print(f"시스템 상태 변경: {STAY} -> {ALIGNING} (목표: 밀봉)")
    return {"status": "ok", "message": "Alignment process for sealing has been started."}

@app.post("/control/open", tags=["Arduino Control"])
async def start_opening():
    """개봉을 위한 정렬 프로세스를 시작합니다."""
    global SYSTEM_STATE, TARGET_ACTION
    with state_lock:
        if SYSTEM_STATE != STAY:
            return JSONResponse(status_code=409, content={"status": "error", "message": f"System is busy with '{SYSTEM_STATE}'"})
        SYSTEM_STATE = ALIGNING
        TARGET_ACTION = 'O'
        print(f"시스템 상태 변경: {STAY} -> {ALIGNING} (목표: 개봉)")
    return {"status": "ok", "message": "Alignment process for opening has been started."}

@app.post("/control/home", tags=["Arduino Control"])
async def return_to_home():
    """초기 위치 복귀 ('H') 신호를 아두이노에 전송합니다."""
    success, message = send_serial_command('H')
    if success:
        return {"status": "ok", "message": message}
    return JSONResponse(status_code=500, content={"status": "error", "message": message})
    
@app.post("/control/stop", tags=["Arduino Control"])
async def emergency_stop():
    """긴급 정지 ('E') 신호를 아두이노에 전송하고 시스템 상태를 초기화합니다."""
    global SYSTEM_STATE, TARGET_ACTION
    with state_lock:
        SYSTEM_STATE = STAY
        TARGET_ACTION = None
    success, message = send_serial_command('E')
    if success:
        return {"status": "ok", "message": f"{message}. System state has been reset to '{STAY}'."}
    return JSONResponse(status_code=500, content={"status": "error", "message": message})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    print(f"클라이언트 연결: {len(clients)}명 접속 중")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.remove(websocket)
        print(f"클라이언트 연결 끊김: {len(clients)}명 접속 중")

# MJPEG 영상 스트리밍 (소비자)
def generate_annotated_frame():
    while True:
        with frame_lock:
            if latest_annotated_frame is None:
                continue
            frame_bytes = latest_annotated_frame
        
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_annotated_frame(), media_type="multipart/x-mixed-replace; boundary=frame")

# React 정적 파일 서빙
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

