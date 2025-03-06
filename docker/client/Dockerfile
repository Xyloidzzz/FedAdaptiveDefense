FROM python:3.10-slim

WORKDIR /app

# copy necessary files
COPY pyproject.toml /app/

# install dependencies
RUN pip install --no-cache-dir -e .

# copy the application code
COPY baseline/ /app/baseline/
COPY main.py /app/
COPY checkpoints /app/checkpoints
COPY results /app/results

# entrypoint script executable
COPY docker/client/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]