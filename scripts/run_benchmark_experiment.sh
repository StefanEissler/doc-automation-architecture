#!/bin/bash
#SBATCH --job-name=llm_pilot
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=pilot_output_%j.out
#SBATCH --error=pilot_error_%j.err

echo "Starte Pilot-Experiment auf bwUniCluster 3.0"

# Workspace finden und validieren
WS_DIR=$(ws_find llm_data)
if [ -z "$WS_DIR" ]; then
    echo "Kritischer Fehler: Workspace 'llm_data' nicht gefunden!"
    exit 1
fi

export ENROOT_DATA_PATH="$WS_DIR/enroot_data"
export ENROOT_CACHE_PATH="$WS_DIR/enroot_cache"
export OLLAMA_MODELS="$WS_DIR/ollama_models"
export XDG_DATA_HOME="$WS_DIR"

mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$OLLAMA_MODELS"

# Ollama Container importieren
if [ ! -f "$WS_DIR/ollama.sqsh" ]; then
    echo "Importiere Ollama Container..."
    cd "$WS_DIR" || exit 1
    enroot import --output ollama.sqsh docker://ollama/ollama
    enroot create --name ollama_container ollama.sqsh
fi

# Ollama Daemon im Hintergrund starten (ohne 'bash -c')
echo "Starte Ollama Daemon..."
enroot start -m "$OLLAMA_MODELS:/root/.ollama" --rw ollama_container serve &

sleep 20

# Ins Projektverzeichnis wechseln
cd "$HOME/doc-automation-architecture" || exit 1

# Benchmark-Pipeline ausführen
echo "Starte Benchmark über die Architektur-Bedingungen..."
uv run python main.py --condition all --complexity all --limit 20 --provider ollama --model llama3

echo "Pilot-Experiment erfolgreich beendet."