from fastapi import APIRouter, HTTPException, status

from api.schemas.auth import LoginRequest, LoginResponse, UserResponse

from auth.auth import authenticate


router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):

    user = authenticate(request.username, request.password)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Username or Password")

    return LoginResponse(
        message="Login Successful",
        user=UserResponse(
            id=user["id"],
            username=user["username"],
            role=user["role"],
            department=user["department"],
            team=user["team"]
        )
    )