#!/usr/bin/env bash

GPU="${1}"
JUPYTER_PORT="${2}"
TENSORBOARD_PORT="${3}"

if [ "${#}" -ne 3 ]; then
  echo "Run this script as: ${0} <GPU-ID> <JUPYTER-PORT> <TENSORBOARD-PORT>"
  exit 1
fi

docker run --detach \
           --name "ogb-lsc-$(whoami)" \
	   --volume "${HOME}/ogb-kdd21:/app" \
	   --gpus "device=${GPU}" \
	   --ipc=host \
	   --port "${JUPYTER_PORT}:8888" \
	   --port "${TENSORBOARD_PORT}:6006" \
	   ogblsc:latest /bin/bash -c "trap : TERM INT; sleep infinity & wait"
