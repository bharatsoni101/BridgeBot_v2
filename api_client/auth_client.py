import requests


FASTAPI_BASE_URL = "http://127.0.0.1:8000"


def login_user(username: str, password: str):

    url = f"{FASTAPI_BASE_URL}/api/auth/login"

    payload = {
        "username": username,
        "password": password
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code == 401:
            return None

        response.raise_for_status()

    except requests.exceptions.RequestException as ex:
        raise RuntimeError(
            f"Unable to connect to FastAPI server: {ex}"
        ) from ex