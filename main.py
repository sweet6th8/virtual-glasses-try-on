from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from model import process_frame, load_glasses  

app = FastAPI()
glasses = load_glasses("glasses7.jpeg")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    result = process_frame(frame, glasses)
    _, buffer = cv2.imencode('.jpg', result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
