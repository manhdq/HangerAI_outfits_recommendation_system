#!/bin/zsh
uvicorn lib.api:app --reload --host 127.0.0.1 --port 3000