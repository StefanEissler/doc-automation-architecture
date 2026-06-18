#!/bin/bash
#SBATCH --job-name=llm_pilot
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=pilot_output_%j.out
#SBATCH --error=pilot_error_%j.err

echo "Starte Pilot-Experiment auf bwUniCluster 3.0"

# Dateisystem und Workspaces konfigurieren
# Dokumentierter Workaround für Workspaces.
# Dies stellt sicher, dass entpackte Container nicht das Home-Quota sprengen.
WS_DIR=$(ws_find llm_data)
export XDG_DATA_HOME=$WS_DIR
export ENROOT_DATA_PATH=$WS_DIR/enroot

# Ollama Container vorbereiten
# Importiert das Image von DockerHub in eine .sqsh-Datei und entpackt es.
if [ ! -f "$WS_DIR/ollama.sqsh" ]; then
    echo "Importiere Ollama Container..."
    cd $WS_DIR
    enroot import --output ollama.sqsh docker://ollama/ollama
    enroot create --name ollama_container ollama.sqsh
fi

# Ollama Server im Hintergrund starten
# Wir starten den Container im Lese-/Schreibmodus (--rw).
# Durch das Mounten (-m) eines lokalen Verzeichnisses wird sichergestellt,
# dass die heruntergeladenen Llama-Modelle persistent im Workspace liegen.
export OLLAMA_MODELS="$WS_DIR/ollama_models"
mkdir -p $OLLAMA_MODELS

echo "Starte Ollama Daemon..."
enroot start -m $OLLAMA_MODELS:/root/.ollama --rw ollama_container bash -c "ollama serve" &

sleep 15

# Llama 3 Modell vorab in den Cache laden, um Timeouts im Python-Code zu verhindern.
enroot start --rw ollama_container bash -c "ollama pull llama3"

# Python Benchmark-Pipeline ausführen
echo "Aktiviere Python Environment..."
source $WS_DIR/venv/bin/activate

echo "Starte Benchmark über die Architektur-Bedingungen..."
# Das Skript greift via localhost:11434 auf den Hintergrund-Container zu.
python main.py --condition all --complexity all --limit 20 --provider ollama --model llama3

echo "Pilot-Experiment erfolgreich beendet."