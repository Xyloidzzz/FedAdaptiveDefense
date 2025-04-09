# FedAdaptiveDefense

## Installation

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
```

## Viewing Results

### wandb Integration

To create a wandb account, go to [wandb](https://wandb.ai/) and sign up. After signing up, you can create a new project and get your API key. You can set the API key in your environment variables using the following command:

```bash
wandb login
```

This command will prompt you to enter your API key. After entering the API key, you can run the federated learning environment with wandb integration as normal.

To view the results on wandb, you will be given a link at the end of the simulation. You can also view the results by going to your wandb account and selecting the project you created.

### Results Visualization

When you run the federated learning environment, the results will be saved in the `results` directory. You can view the results by opening the `results` directory and checking the files inside.

If wandb does not work or you don't wish to create an account, you can use the results.py script to visualize the results. To do this, run the following command:

```bash
python results.py
```
