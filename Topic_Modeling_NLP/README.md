### Project Summary

__Author-Topic Latent Dirichlet Allocation__: 

The notebook [Latent_Dirichlet_Allocation.ipynb](Latent_Dirichlet_Allocation.ipynb)  uses the corpus, dictionary and other data built in [Text_Mining_Processing.ipynb](Text_Mining_Processing.ipynb) to train a Author-Topic Latent Dirichlet Allocation (LDA) model. LDA model is a topic modelling technique. In a text corpus, each document is associated with a multinomial distribution over topics, and each topic is associated with a multinomial distribution over words. Here, Dirichlet distributions are used as prior distributions. The author topic model extends LDA to include authorship information.  For the author-topic model, each author is associated with a multonimial distribution over topics.

__Text mining and processing__: 

The notebook [Text_Mining_Processing.ipynb](Text_Mining_Processing.ipynb) execute two parts (i) text mining  and (ii) text processing:

- __Text mining__: Documents are created out of text extracted from Wikipedia pages using the wikipedia api and [`Wikipedia-API`](https://pypi.org/project/Wikipedia-API/). This is done by using home built python script [Fetch_Wiki_data.py](Fetch_Wiki_data.py).

- __Text processing__: The text of the documents are then processed using [spaCy](https://spacy.io/) library. That includes
    - tokenization of words
    - cleaning-up
    - lemmatization
    - POS tagging

In the end a dictionary is created, mapping words to numerical ids, and the documents are converted to a bag-of-words format


Bash commands one can use for troubleshooting various issues and setting up environment

```bash
make                  # builds NLP_env + installs everything else
source NLP_env/bin/activate
# work...
deactivate
make clean            # remove caches only
make distclean        # remove entire venv too
```

```bash
source NLP_env/bin/activate # To activate environment  again
python -m spacy download en_core_web_sm  # to download seprately
```

Instead of this
```bash
python -m spacy download en_core_web_sm
```
try this for better uses

```bash
pip install \
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

```bash
pip install --upgrade pip setuptools wheel
```

```bash
pip install ipywidgets notebook jupyterlab
```

or  classic Jupyter:

```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```


```bash
git rm -r --cached NLP_env
git commit -m "Stop tracking virtual environment"
git status --ignored
```

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```


- Install:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install Wikipedia-API
```
