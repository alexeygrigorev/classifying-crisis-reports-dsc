# Growing Instability: Classifying Crisis Reports

This is a top 10 solution to the ["Growing Instability: Classifying Crisis Reports"](https://www.datasciencechallenge.org/challenges/2/growing-instability) challenge from datasciencechallenge.org

The solution outline:

- Use only the data from 2005 onwards for training
- Discard the labels with less than 30 examples
- Vectorize the texts, apply TF-IDF weights
- For each label train an SVM
- Use out-of-fold predictions for train for selecting the best threshold 
- For topics present in test and not present in train do a simple keyword matching
