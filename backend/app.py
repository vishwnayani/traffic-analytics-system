from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

import os
import shutil

from backend.inference import get_model

from backend.tracker import process_video

from backend.config import (
    UPLOAD_DIR
)

app = FastAPI(
    title="Traffic Analytics API"
)

model = get_model()


@app.get("/")
def home():

    return {

        "message":
        "Traffic Analytics API Running"

    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    save_path = os.path.join(

        UPLOAD_DIR,

        file.filename

    )

    with open(
        save_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = process_video(

        save_path,

        model

    )

    return result