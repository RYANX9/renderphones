#!/usr/bin/env bash
uvicorn fapi:app --host 0.0.0.0 --port $PORT