.PHONY: train eval app test clean

train:
	python scripts/train.py

eval:
	python scripts/eval.py

app:
	streamlit run streamlit_app.py

test:
	pytest -q

clean:
	-find . -type d -name "__pycache__" -exec rm -r {} + || true
	-find . -name "*.pyc" -delete || true
