from datetime import datetime, timedelta
from jose import JWTError, jwt

# SECRET_KEY
# Algorithm
# Expiration time

SECRET_KEY = "d6638d9a6f19c9239acd5f804e6fa9d88fe29f746161f197c273468827e99abb"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def create_access_token(data: dict):
    to_encode = data.copy()

    expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt