setup:
\tpython -m pip install -r requirements.txt
\tpython -m spacy download en_core_web_sm || true

eda:
\tpython scripts/run_pipeline.py --stage eda --config configs/config.yaml

extract:
\tpython scripts/run_extraction.py --config configs/config.yaml

summarize:
\tpython scripts/run_summarization.py --config configs/config.yaml

agent:
\tpython scripts/run_agent_demo.py --config configs/config.yaml

test:
\tpytest -q

format:
\truff check --fix . || true
\tblack . || true

preprocess-reuters:
\tpython scripts/run_pipeline.py --dataset reuters

preprocess-amazon:
\tpython scripts/run_pipeline.py --dataset amazon_polarity

preprocess:
\tpython scripts/run_pipeline.py --config configs/config.yaml
