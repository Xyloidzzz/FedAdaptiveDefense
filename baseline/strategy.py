

from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg, Krum, FedMedian, FedTrimmedAvg

import torch
import json 
import wandb
from datetime import datetime

from .task import Net, set_weights

class CustomFedAvg(FedAvg):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(project="flower-simulation-baselines", name=f"custom-FedAvg-{name}")

    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        #initiate model
        model = Net()
        set_weights(model, ndarrays)

        # save global model in standard pytorch way
        torch.save(model.state_dict(), f"checkpoints/global_model_round_{server_round}")

        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self, server_round, parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        results = {"loss": loss, **metrics}
        
        self.results_to_save[server_round] = results

        #save metrics as json
        with open("results/results.json", 'w') as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log to wandb
        wandb.log(results, step=server_round)

        return loss, metrics
    
    
class CustomKrum(Krum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(project="flower-simulation-baselines", name=f"custom-Krum-{name}")

    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        #initiate model
        model = Net()
        set_weights(model, ndarrays)

        # save global model in standard pytorch way
        torch.save(model.state_dict(), f"checkpoints/global_model_round_{server_round}")

        return parameters_aggregated, metrics_aggregated

    def evaluate(self, server_round, parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        results = {"loss": loss, **metrics}
        
        self.results_to_save[server_round] = results

        #save metrics as json
        with open("results/results.json", 'w') as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log to wandb
        wandb.log(results, step=server_round)

        return loss, metrics


class CustomFedMedian(FedMedian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(project="flower-simulation-baselines", name=f"custom-FedMedian-{name}")

    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        #initiate model
        model = Net()
        set_weights(model, ndarrays)

        # save global model in standard pytorch way
        torch.save(model.state_dict(), f"checkpoints/global_model_round_{server_round}")

        return parameters_aggregated, metrics_aggregated

    def evaluate(self, server_round, parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        results = {"loss": loss, **metrics}
        
        self.results_to_save[server_round] = results

        #save metrics as json
        with open("results/results.json", 'w') as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log to wandb
        wandb.log(results, step=server_round)

        return loss, metrics
    

class CustomFedTrimmedAvg(FedTrimmedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(project="flower-simulation-baselines", name=f"custom-FedTrimmedAvg-{name}")

    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        #initiate model
        model = Net()
        set_weights(model, ndarrays)

        # save global model in standard pytorch way
        torch.save(model.state_dict(), f"checkpoints/global_model_round_{server_round}")

        return parameters_aggregated, metrics_aggregated

    def evaluate(self, server_round, parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        results = {"loss": loss, **metrics}
        
        self.results_to_save[server_round] = results

        #save metrics as json
        with open("results/results.json", 'w') as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log to wandb
        wandb.log(results, step=server_round)

        return loss, metrics
