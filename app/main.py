import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI

from . import models
from .database import engine, get_db
from .routers import user, auth, apparel

models.Base.metadata.create_all(bind=engine)


app = FastAPI()


app.include_router(user.router)
app.include_router(auth.router)
app.include_router(apparel.router)

@app.get("/")
def root():
    return {"message": "Hello World"}