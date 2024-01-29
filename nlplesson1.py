# coding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import Tokenixer
from tensorflow.keras.preprocessing.Text import Tokenixer
from tensorflow.keras.preprocessing.text import Tokenizer
sentences = [
'I love my dog',
'I love my cat'
]
tokenizer = Tokenizer()   # create instances
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)



