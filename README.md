# n-gram_python
A python solution for n-gram method in NLP.

## Training

Put your training data in the 'data/' directory (or anywhere you like), and you can train a trigram model through:

```bash
python train.py -n 3 -f data/train_set.txt
```

Token counts will be generated in the form of json files in the 'n_gram_bank/' directory.

## Testing

Put your testing data in the 'data/' directory (or anywhere you like), and you use the trained trigram model to test through:

```bash
python test.py -n 3 -f data/test_set.txt
```

## Discounting method

Different discounting methods are provided. Now includes:

* Good Turing Discounting: 'turing' (Default)
* Gumbel Discounting: 'gumbel'

Take Truing Discounting as an example:

```bash
python train.py -n 3 -f data/train_set.txt -m turing
```

## Instant testing

After the model is trained, you can instantly test your sentence through the '-inst' arg.

Note that words should be connected by bars, and any punctuation or capital letter should not be included.


```bash
python test.py -n 2 -inst every-day-he-gets-up-at-six-goes-jogging-and-eats-breakfast-at-seven
```

which outputs:

```
PPL = PPL = 581.36260
```

## Interesting facts
Through the instant feedback command, you can see how a right-ordered scentence gets a lower probability when it's scrambled:

```bash
python test.py -n 2 -inst mother-always-say-an-apple-a-day-keeps-the-doctor-away
```

gets the result:

```
PPL = 1122.59597
```

```bash
python test.py -n 2 -inst apple-always-say-an-doctor-a-day-keeps-the-mother-away
```

gets the result:

```
PPL = 1264.10669
```

```bash
python test.py -n 2 -inst always-away-mother-an-apple-day-doctor-a-keeps-the-say
```

gets the result:

```
PPL = 1747.99034
```

As the sentence gets more confused, PPL increases.