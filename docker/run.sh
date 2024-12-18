#!/bin/bash

IMAGE_NAME="mmdetection3d:dev-tvg"
CONTAINER_NAME="mmdetection3d-dev-tvg"

HOST_DATA_DIR="/media/tvg/my_label/datasets/nuscenes"
CONTAINER_DATA_DIR="/workspace/data/nuscenes"

HOST_WORK_DIR="/home/tvg/Projects/robust-bev/mmdetection3d"
CONTAINER_WORK_DIR="/workspace"

SHM_SIZE="64g"
GPU_DEVICES="all"

# if docker ps -a --filter name="${CONTAINER_NAME}" --format "{{.Names}}" | grep -w "${CONTAINER_NAME}" > /dev/null; then
#     read -p "Container '${CONTAINER_NAME}' exists. Do you want to remove it? (y/n) " choice
#     if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
#         docker rm -f "${CONTAINER_NAME}"
#     else
#         read -p "Enter the image name to commit the container to [${IMAGE_NAME}]: " image_name
#         image_name=${image_name:-${IMAGE_NAME}}
#         docker commit "${CONTAINER_NAME}" "${image_name}"
#         docker rm -f "${CONTAINER_NAME}"
#     fi
# fi

docker run -it \
    --gpus "device=${GPU_DEVICES}" \
    --name "${CONTAINER_NAME}" \
    --shm-size=${SHM_SIZE} \
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

# TODO: Add this back in after data preprocessing is done
# -v "${HOST_DATA_DIR}:/data:ro" \