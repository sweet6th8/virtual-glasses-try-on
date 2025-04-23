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
    allow_origins=["*"],  # hoặc ["http://localhost:3000"]
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

            # B1: Decode ảnh base64
            header, encoded = data.split(",", 1)
            img_data = base64.b64decode(encoded)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # B2: Xử lý AI hoặc giả lập overlay kính
            frame = process_frame(frame, glasses)

            # B3: Encode lại và gửi về
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{encoded_image}")
    except Exception as e:
        print("WebSocket Error:", e)
        await websocket.close()
