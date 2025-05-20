from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from model import process_frame, glasses
import json
import requests

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
            print("Dữ liệu nhận được:", data[:100])
            try:
                # Parse JSON từ FE
                payload = json.loads(data)
                image_b64 = payload.get("image")
                glasses_url = payload.get("glasses_url")
                if not image_b64 or not glasses_url:
                    print("Thiếu image hoặc glasses_url")
                    continue

                # Giải mã frame
                img_data = base64.b64decode(image_b64.split(",")[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)
                if frame is None:
                    print("Không giải mã được frame, bỏ qua.")
                    continue

                # Tải ảnh kính từ URL FE gửi lên
                response = requests.get(glasses_url)
                glasses_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
                glasses_img = cv2.imdecode(glasses_arr, cv2.IMREAD_UNCHANGED)
                if glasses_img is None:
                    print("Không tải được ảnh kính từ URL:", glasses_url)
                    continue

                processed_frame = process_frame(frame, glasses_img)
                _, buffer = cv2.imencode('.png', processed_frame)
                encoded = base64.b64encode(buffer).decode()
                await websocket.send_text(f"data:image/png;base64,{encoded}")

            except Exception as e:
                print("Lỗi xử lý:", e)
                continue

    except Exception as e:
        print(f"Lỗi WebSocket: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
