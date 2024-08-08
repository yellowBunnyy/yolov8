### Commands

`> docker build -t [img_name] --build-arg GPU_ON=True .`<br>
`> docker run -it --name dv_8 -p 8000:8000 --gpus all [img_name]`