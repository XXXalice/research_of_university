from gensim.models import KeyedVectors

model_data = 'wikipedia.model.bin'

model = KeyedVectors.load_word2vec_format('./data/' + model_data, binary=True)
not_similar_words = model.most_similar()
print(not_similar_words)
print(len(not_similar_words))
