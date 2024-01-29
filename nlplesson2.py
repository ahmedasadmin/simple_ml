# coding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
sentences = [
'I love my dog',
'I love my cat'
]
tokenizer = Tokenizer(num_words=100, oov_token="<oov>")   # create instances
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)


sequences = tokenizer.texts_to_sequences(sentences)
test_data = [
	'i really loves my dog',
	'my dog loves manatee'
	]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)


## padding using tensorflow 


from tensorflow.keras.preprocessing.sequence import pad_sequences 

padded = pad_sequences(sequences, padding='post',truncating='post', maxlen=5)
print("---------------------------------")
print(padded)




















