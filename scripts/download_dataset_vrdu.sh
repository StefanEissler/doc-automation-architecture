#!/bin/bash

echo "Klone VRDU Dataset..."

TARGET_DIR="data/vrdu"
REPO_URL="https://github.com/google-research-datasets/vrdu.git"

if [ ! -d "$TARGET_DIR" ]; then
  git clone --no-checkout --depth 1 --filter=blob:none "$REPO_URL" "$TARGET_DIR"
  
  cd "$TARGET_DIR" || exit
  
  git sparse-checkout set \
    registration-form/main \
    registration-form/few_shot-splits \
    ad-buy-form/main \
    ad-buy-form/few_shot-splits

  git sparse-checkout add ad-buy-form/few_shot-splits

  git checkout main

  gunzip registration-form/main/dataset.jsonl.gz
  gunzip ad-buy-form/main/dataset.jsonl.gz
  
  cd ../../..
else
  echo "VRDU bereits vorhanden."
fi

echo "VRDU Dataset herunterladen"