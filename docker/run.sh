#!/bin/bash

IMAGE_NAME="tanmay4269/ifss:ritm"
CONTAINER_NAME="ifss-ritm"

HOST_WORK_DIR="/home/tvg/Documents/Projects/iFSS"
CONTAINER_WORK_DIR="/workspace"

docker run -it \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${HOST_WORK_DIR}:${CONTAINER_WORK_DIR}" \
    -w "${CONTAINER_WORK_DIR}" \
    --network host \
    --ipc=host \
    ${IMAGE_NAME} 

read -p "Do you want to commit the container? (y/n) " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    read -p "Enter the image name to commit the container to [${IMAGE_NAME}]: " image_name
    image_name=${image_name:-${IMAGE_NAME}}
    docker commit "${CONTAINER_NAME}" "${image_name}"
fi

docker rm  -f "${CONTAINER_NAME}"