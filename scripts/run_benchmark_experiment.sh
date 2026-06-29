#!/bin/bash
#SBATCH --job-name=llm_pilot
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=pilot_output_%j.out
#SBATCH --error=pilot_error_%j.err

echo "Starting with bwUniCluster Configruation"

export PYTHONPATH=""
export PYTHONNOUSERSITE=1
source /opt/bwhpc/common/etc/easybuild/enable_eb_modules
module load Python/3.12.3-GCCcore-13.3.0

echo "Cleaning old Processes"
pkill -f "ollama serve" || true
pkill -f "ollama pull" || true
sleep 3

WS_DIR=$(ws_find llm_data)
if [ -z "$WS_DIR" ]; then
    echo "Critical error llm_data workspace not found"
    exit 1
fi

export ENROOT_DATA_PATH="$WS_DIR/enroot_data"
export ENROOT_CACHE_PATH="$WS_DIR/enroot_cache"
export OLLAMA_MODELS="$WS_DIR/ollama_models"
export XDG_DATA_HOME="$WS_DIR"

mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$OLLAMA_MODELS" "$WS_DIR/log"

if [ ! -f "$WS_DIR/ollama.sqsh" ]; then
    echo "Importing Ollama Container"
    cd "$WS_DIR" || exit 1
    enroot import --output ollama.sqsh docker://ollama/ollama
    enroot create --name ollama_container ollama.sqsh
fi

echo "Strating Ollama Daemon"
enroot start -m "$OLLAMA_MODELS:/root/.ollama" --rw ollama_container serve > "$WS_DIR/log/ollama_daemon.log" 2>&1 &

echo "Waiting for Ollama API"
timeout=120
elapsed=0
while ! curl -s http://localhost:11434/api/version > /dev/null; do
    sleep 5
    elapsed=$((elapsed+5))
    if [ "$elapsed" -ge "$timeout" ]; then
        echo "CRITICAL ERROR: API not startet within $timeout seconds"
        exit 1
    fi
done
echo "Ollama API is reachable"

cd "$HOME/doc-automation-architecture" || exit 1

echo "Starting the Benchmark Experiment:"
uv run python -m main --experiment A --model llama3.3

echo "Experiment was succesfull"