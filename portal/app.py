from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import sys
import uuid
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.visual_model import get_model
from models.fusion import fuse_scores
from utils.heatmap import generate_heatmap, get_transform
from utils.audit_log import log_entry, get_logs, verify_log_chain


app = FastAPI(title="Deepfake Detection Portal", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
visual_model = None
transform = get_transform()


@app.on_event("startup")
async def load_models():
    global visual_model
    print("📦 Loading model...")
    visual_model = get_model(DEVICE)
    visual_model.load_state_dict(
        torch.load('models/visual_best.pth', map_location=DEVICE)
    )
    visual_model.eval()
    print("🚀 Server ready!")


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/verify/image")
async def verify_image(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    filepath = f"uploads/{file_id}_{file.filename}"
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Heatmap + prediction
        heatmap, fake_prob = generate_heatmap(visual_model, filepath, DEVICE)
        
        # Save heatmap
        heatmap_path = f"outputs/{file_id}_heatmap.jpg"
        Image.fromarray(heatmap).save(heatmap_path)
        
        # Fusion
        result = fuse_scores(fake_prob, 0.0)
        result['heatmap_url'] = f"/outputs/{file_id}_heatmap.jpg"
        result['file_id'] = file_id
        
        # Audit log
        log_entry(result, filepath)
        
        return JSONResponse(result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs")
async def logs():
    return {"logs": get_logs(), "chain_status": verify_log_chain()}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("portal/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)