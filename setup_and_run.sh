#!/bin/bash

pip install -r requirements.txt
python -m spacy download en_core_web_trf
python -m spacy download de_dep_news_trf

cd src
echo "================="
echo "|It's taco time!|"
echo "================="
python main.py -output_dir output/
