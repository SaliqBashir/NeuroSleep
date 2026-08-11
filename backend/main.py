from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from backend.inference import predict_sleep_stages

app = FastAPI(title="NeuroSleep API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".edf"):
        raise HTTPException(
            status_code=400, detail="Only .edf files are supported")

    # Save the file temporarily
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.edf")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference
        stages = predict_sleep_stages(file_path)

        # Calculate summary statistics
        total_epochs = len(stages)
        summary = {
            "Wake": stages.count("Wake"),
            "N1": stages.count("N1"),
            "N2": stages.count("N2"),
            "N3": stages.count("N3"),
            "REM": stages.count("REM"),
            "total_epochs": total_epochs,
            "total_minutes": total_epochs * 30 / 60
        }

        return {
            "status": "success",
            "filename": file.filename,
            "stages": stages,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
