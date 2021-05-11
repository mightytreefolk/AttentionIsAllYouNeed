# AttentionIsAllYouNeed
CS 547 Semester Project

Data can be found at: https://nlp.stanford.edu/projects/nmt/

for the  code to run the data must be placed at the same directory level as src with the following data structure:

```
|-- src
├── data
    ├── test_data
    │   ├── newstest2014.de
    │   └── newstest2014.en
    ├── train_data
    │   ├── train.de
    │   └── train.en
    └── vocab_data
        ├── vocab.50K.de
        └── vocab.50K.en
```

To set up virtualenv use pipenv

Word vectorization algorithm: https://github.com/stanfordnlp/GloVe
This algorithm was used to vectorize the training datasets based on vocabulary. 
https://gist.github.com/michaelchughes/85287f1c6f6440c060c3d86b4e7d764b

