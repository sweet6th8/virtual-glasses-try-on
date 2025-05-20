from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from model import process_frame, glasses

app = FastAPI()

# Cho phép React gọi từ domain khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print("Dữ liệu nhận được:", data[:50])
            if not data or ',' not in data:
                print("Dữ liệu không hợp lệ, bỏ qua.")
                continue
            try:
                img_data = base64.b64decode(data.split(",")[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)  # Lật ngang ảnh từ webcam browser
                cv2.imwrite("test_frame.png", frame)
                if frame is None:
                    print("Không giải mã được frame, bỏ qua.")
                    continue
                print("Kích thước frame nhận được:", frame.shape)
            except Exception as e:
                print("Lỗi giải mã ảnh:", e)
                continue

            processed_frame = process_frame(frame, glasses)
            _, buffer = cv2.imencode('.png', processed_frame)
            encoded = base64.b64encode(buffer).decode()
            await websocket.send_text(f"data:image/png;base64,{encoded}")

    except Exception as e:
        print(f"Lỗi WebSocket: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
