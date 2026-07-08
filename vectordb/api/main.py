from contextlib import asynccontextmanager

from fastapi import FastAPI

from vectordb.api.routes import router
from vectordb.api.state import get_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_registry()
    yield
    get_registry().close_all()


app = FastAPI(title="vector-db", lifespan=lifespan)
app.include_router(router)
