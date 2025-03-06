import subprocess

if __name__ == "__main__":
    subprocess.run(["flwr", "run", "."], check=True)
