.PHONY: demo install clean ui quick

install:
	python -m pip install -r requirements.txt

demo: install
	python run_all.py

quick:
	python run_all.py --quick

ui:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/*.parquet data/*.npy artifacts/*.pkl artifacts/*.parquet artifacts/figures/*.png
