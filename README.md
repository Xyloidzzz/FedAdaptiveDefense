# FedAdaptiveDefense

## Running the Federated Learning Environment

To run the federated learning environment using Docker, you can use the provided `docker-compose.yml` file. This will set up the server and multiple clients.

1. Build the containers.:

   ```bash
   docker-compose build
   ```

2. Start the containers:

   ```bash
   docker-compose up
   ```

3. The server will be running and ready to accept client connections.

## Configuration

The configuration for the federated learning setup can be found in the `pyproject.toml` file. You can adjust parameters such as the number of server rounds and the fraction of clients to participate in each round.

## Checkpoints and Results

The trained model checkpoints will be saved in the `checkpoints/` directory, and the results of the training will be stored in the `results/` directory.

<!-- ## Installation

Please create a virtual environment and activate it before installing the required packages. You can create a virtual environment using the following command:

```bash
python -m venv venv
```

Install the required packages using the following command:

```bash
pip install -e .
```

## Run Federated Learning Environment

To run the federated learning environment, use the following command:

```bash
flwr run .
``` -->
