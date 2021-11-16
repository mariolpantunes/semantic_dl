# semantic_dl


# Instalation
- This commands are run on the git root folder
- make sure to have c++11 installed as fastText needs it.
- compile fasttext 
    - $ wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
    - $ unzip v0.9.2.zip
    - $ cd fastText-0.9.2
    - $ make 
    - move the binary file "fasttext" to the fasttext folder (mv fastText-0.9.2/fasttext fasttext/) 
- install python3-dev
    -sudo apt install python3-dev
- Download dataset for similarity https://raw.githubusercontent.com/AlexGrinch/ro_sgns/master/datasets/rg65.csv
    - put it in dataset/train_sim/
- Download pretrained fasttext https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
    - put it in fasttext/pre-trained/
    - rename it pretrained.vec
- Download pretrained glove https://nlp.stanford.edu/data/glove.6B.zip
    - put it in glove/pre-trained/
- Install python libraries
    - pip install -r requirements.txt
- Download spacy pretrained model
    - python -m spacy download en_core_web_lg