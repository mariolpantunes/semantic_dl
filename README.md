# semantic_dl


# Instalation
- This commands are run on the git root folder
- make sure to have c++11 installed as fastText needs it.
- compile fasttext 
    - `wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip`
    - `unzip v0.9.2.zip`
    - `cd fastText-0.9.2`
    - `make`
    - move the binary file `fasttext` to the fasttext folder
        - `mv fastText-0.9.2/fasttext fasttext/`
- install python3-dev
    - `sudo apt install python3-dev` (for debain based systems)
- Download dataset for similarity training https://raw.githubusercontent.com/AlexGrinch/ro_sgns/master/datasets/rg65.csv
    - put it in `dataset/train_sim/`
- Download constrained corpus https://www.kaggle.com/datasets/mantunes/semantic-corpus-from-web-search-snippets
    - put the uncompressed files (`.csv` format) in `dataset/train/`
- Download dataset for similarity evaluation (IoT) https://www.kaggle.com/datasets/mantunes/semantic-iot
    - put it in `dataset/test` with the name `en-mc-30.csv`
- Download dataset for similarity evaluation (MC) https://www.kaggle.com/datasets/mantunes/millercharles
    - put it in `dataset/test` with the name `en-iot-30.csv`
- Download pretrained fasttext https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
    - put it in `fasttext/pre-trained/`
    - rename it `pretrained.vec`
- Download pretrained glove https://nlp.stanford.edu/data/glove.6B.zip
    - put it in `glove/pre-trained/`
- Install python libraries
    - `pip install -r requirements.txt`

## Authors

* **Mário Antunes** - [rgtzths](https://github.com/mariolpantunes)
* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

##Citation

Teixeira, Rafael & Antunes, Mário & Gomes, Diogo & Aguiar, Rui. (2022). Comparison of Semantic Similarity Models on Constrained Scenarios. Information Systems Frontiers. 10.1007/s10796-022-10350-w. 
