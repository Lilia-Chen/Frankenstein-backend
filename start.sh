#!/bin/bash

# Load environment variables from .env and start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
