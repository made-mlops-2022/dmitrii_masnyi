# Homework 3  
## Start 
In the directory with ```docker-compose.yml``` run 
```
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose up --build
```
## Stop
In the directory with ```docker-compose.yml``` run 
```
docker-compose down
```
