#!/bin/bash
set -e

# Generate a random partition ID (0-9)
PARTITION_ID=$((RANDOM % 10))
echo "Starting Flower client with partition ID: $PARTITION_ID"

cd /app
python -c "
import uuid
from flwr.client import start_client
from baseline.client_app import client_fn
from flwr.common import Context

# Create a properly initialized Context object
context = Context(
    run_id=str(uuid.uuid4()),  # Generate a unique run ID
    node_id=f'client-{$PARTITION_ID}',  # Client node ID with partition
    node_config={'partition-id': $PARTITION_ID, 'num-partitions': 10},
    state={},  # Empty initial state
    run_config={'local-epochs': 1}  # Your run config
)

# Start the client with the proper context
start_client(server_address='$SERVER_URL', client=client_fn(context))
"