### Environments

Fill the jupyter_workspace/.env, model_hosting_api/.env, and .postgres/.env with whatever values you see fit (consistent ofc).

### Running

Just have docker installed and running and execute:

    docker-compose up -d

from the project root.

### Jupyter Notebook

You can acess the Jupyter Notebook from the host browser by navigating to:

    http://localhost:8888/tree

### Databse

To psql into the db you can start an interactive Bash shell with:

    docker exec -it postgres bash