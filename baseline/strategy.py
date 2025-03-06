

from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg, Krum, FedMedian, FedTrimmedAvg

import torch
import json 
import os
from datetime import datetime

from .task import Net, set_weights

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# log to wandb if available and enabled
def safe_wandb_log(metrics, step):
    if WANDB_AVAILABLE and os.environ.get("WANDB_MODE") != "disabled":
        try:
            if wandb.run is None:
                return
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"Warning: wandb logging failed: {e}")

# initialize wandb if available and not disabled
def safe_wandb_init(name_prefix):
    if not WANDB_AVAILABLE or os.environ.get("WANDB_MODE") == "disabled":
        return
    
    try:
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(project="flower-simulation-baselines", name=f"{name_prefix}-{name}", mode="offline")
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")


class CustomFedAvg(FedAvg):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.results_to_save = {}
        
        safe_wandb_init("custom-FedAvg")

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
        safe_wandb_log(results, step=server_round)

        return loss, metrics
    
    
class CustomKrum(Krum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        safe_wandb_init("custom-Krum")

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
        safe_wandb_log(results, step=server_round)

        return loss, metrics


class CustomFedMedian(FedMedian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        safe_wandb_init("custom-FedMedian")

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
        safe_wandb_log(results, step=server_round)

        return loss, metrics
    

class CustomFedTrimmedAvg(FedTrimmedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        safe_wandb_init("custom-FedTrimmedAvg")

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
        safe_wandb_log(results, step=server_round)

        return loss, metrics
