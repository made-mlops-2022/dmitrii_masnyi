### HW2
### Build docker image:
cd to `online_inference/`, then run:
```
docker build --tag dmasny99/online_inference:v2 .   
```

or pull from dockerhub:
```
docker pull dmasny99/online_inference:v1
```

### Run container
```
docker run -p 8000:8000 dmasny99/online_inference:v2
```
Server binds 8000 port on local host by docker configuration.

### Send requests
cd to `online_inference/`, then run:
```
python3 make_requests.py
```
### Test server
firstly, fing an ID of a the contained (you can bind a name during run via flag --name, but I didn't do so)
```
docker ps
```
it will show you all running containers, copy an ID of one you need and then
```
docker exec -it CONT_ID /bin/bash
pytest test_server.py
```