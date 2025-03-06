#!/bin/bash
set -e

echo "Starting Flower server..."
cd /app
python -c "
import uuid
from flwr.server import start_server
from baseline.server_app import server_fn
from flwr.common import Context

# Create a properly initialized Context object
context = Context(
    run_id=str(uuid.uuid4()),  # Generate a unique run ID
    node_id='server-node',     # Server node ID
    node_config={},            # Empty node config for server
    state={},                  # Empty initial state
    run_config={'num-server-rounds': 3, 'fraction-fit': 0.5}  # Your run config
)

# Start the server with the proper context
start_server(server_address='0.0.0.0:8080', server=server_fn(context))
"