from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
	from .routes.api import router
except ImportError:
	from routes.api import router

app = FastAPI(title="Ball-Free Game State Reconstruction API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)