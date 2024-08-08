# yolo_docker
yolo in docker
##Commands 
docker build -t d_v8 --build-arg GPU_ON=True .
docker run -it --name dv_8 -p 8000:8000 --gpus all d_v8