# Support Ticket Classification

IT support ticket classification using a fine-tuned **DistilBERT** model served via FastAPI.

## Architecture

```
main.py                  — FastAPI app (lifespan-managed model, /health, /ticket_support_classification)
src/
  model.py               — DistilBertClassifier (tf.keras.Model), save/load with save_pretrained
  train.py               — Training pipeline (local), shared run_training_pipeline()
  utils.py               — encode_texts, encode_labels, label mapping I/O
  azure_train.py          — Azure ML training entrypoint (standalone imports for remote execution)
  azure_run_train.py      — Submit Azure ML experiment (SDK v1 — legacy, v2 migration planned)
  azure_store_data.py     — Upload data to Azure Blob Storage
  azure_utils.py          — Azure helpers (env-driven secrets, blob client)
  train_conf.yml          — Training hyperparameters
  azure_conf.yml          — Azure config (non-secret values only)
tests/                    — pytest suite (utils, model, API, download)
```

## Requirements

- Python 3.10+
- (Optional) Azure credentials for cloud training — see `.env.example`

## Installation

```console
source install.sh
# or manually:
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Get the Data

```console
python download_data.py
# Produces: all_tickets.csv, subset_tickets.csv
```

**Data variants:**
- `all_tickets.csv` — ~50K samples, 13 unbalanced categories
- `subset_tickets.csv` — ~3K samples, 5 balanced categories

## Train Locally

Configure `src/train_conf.yml` (dataset_path, epochs, batch_size, etc.), then:

```console
python -m src.train
# Produces: my_model/, tokenizer/, label_mapping.json
```

Artifacts saved:
- `my_model/` — TensorFlow SavedModel
- `tokenizer/` — HuggingFace tokenizer (via `save_pretrained`)
- `label_mapping.json` — category id to label name mapping

## Run the API

```console
uvicorn main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` — healthcheck
- `POST /ticket_support_classification` — classify a ticket

```json
// Request
{"message": "My monitor is not working"}

// Response
{"ticket_category": 5, "ticket_category_label": "hardware"}
```

## Docker

```console
docker build -t ticket-classification .
docker run -p 8000:8000 ticket-classification
```

The container runs as non-root with a built-in healthcheck.

## Azure Cloud Training (Legacy SDK v1)

Azure secrets are loaded from environment variables (not committed to the repo).
Copy `.env.example` to `.env` and fill in your credentials:

```
AZURE_SUBSCRIPTION_ID=...
AZURE_STORAGE_ACCOUNT_NAME=...
AZURE_STORAGE_ACCOUNT_KEY=...
```

Then:
```console
python src/azure_store_data.py   # upload data
python src/azure_run_train.py    # submit experiment
```

> **Note:** This uses Azure ML SDK v1 (`azureml-core`/`Estimator`).
> A migration to `azure.ai.ml` v2 is planned as a future improvement.

## Tests

```console
python -m pytest -q --maxfail=1
```

Tests run without network access and without model artifacts.

## CI

GitHub Actions CI (`.github/workflows/ci.yml`) runs on push/PR to `main`:
- Compile checks on all core Python files
- Lint with `ruff`
- Full test suite

## Article

This project has also been discussed [in an article on our website](https://neuronest.net/attribution-des-tickets-aux-equipes-it/).
