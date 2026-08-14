from fastapi import FastAPI

from api.routes.auth import router as auth_router


app = FastAPI(
    title="BridgeBot API",
    description="FastAPI backend for BridgeBot",
    version="1.0.0"
)


app.include_router(auth_router)


@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "application": "BridgeBot"
    }