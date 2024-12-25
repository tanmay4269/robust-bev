#!/bin/bash

IMAGE_NAME="mmdetection3d:dev-tvg"
CONTAINER_NAME="mmdetection3d-dev-tvg"

if [[ "$(hostname)" == "umic-System-Product-Name" ]]; then
    HOST_DATA_DIR="/media/tvg/my_label/datasets/nuscenes"
    HOST_WORK_DIR="/home/tvg/Projects/robust-bev/mmdetection3d"
elif [[ "$(hostname)" == "biplab48gb" ]]; then
    HOST_DATA_DIR="/apps1/tanmay_g/data/nuscenes"
    HOST_WORK_DIR="/apps1/tanmay_g/Projects/robust-bev/mmdetection3d"
else
    echo "Error: This script must be run on the host 'biplab48gb' or 'umic-System-Product-Name'."
    exit 1
fi

CONTAINER_DATA_DIR="/workspace/data/nuscenes"
CONTAINER_WORK_DIR="/workspace"

docker run -it \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}" \
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