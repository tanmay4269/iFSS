#!/bin/bash

IMAGE_NAME="tanmay4269/ifss:latest"
# IMAGE_NAME="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"
CONTAINER_NAME="ifss-ritm"

if [[ $HOSTNAME == "umic-System-Product-Name" ]]; then
    HOST_WORK_DIR="/home/tvg/Projects/iFSS"
    HOST_SAVED_EXPT_DIR="/home/tvg/Projects/iFSS_saved_expts"
elif [[ $HOSTNAME == "biplab48gb" ]]; then
    HOST_WORK_DIR="/apps1/tanmay_g/Projects/iFSS"
    HOST_SAVED_EXPT_DIR="/apps1/tanmay_g/Projects/iFSS_saved_expts"
else
    echo "Unknown host: $HOSTNAME"
    exit 0
fi

CONTAINER_WORK_DIR="/workspace"
CONTAINER_SAVED_EXPT_DIR="/workspace/saved_expts"

docker run -it \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${HOST_WORK_DIR}:${CONTAINER_WORK_DIR}" \
    -v "${HOST_SAVED_EXPT_DIR}:${CONTAINER_SAVED_EXPT_DIR}:ro" \
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