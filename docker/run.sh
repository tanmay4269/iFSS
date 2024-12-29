#!/bin/bash

IMAGE_NAME="tanmay4269/ifss:latest"
# IMAGE_NAME="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
CONTAINER_NAME="ifss-ritm"

if [[ $HOSTNAME == "umic-System-Product-Name" ]]; then
    HOST_WORK_DIR="/home/tvg/Projects/iFSS"
else
    HOST_WORK_DIR="/home/tvg/Documents/Projects/iFSS"
fi

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