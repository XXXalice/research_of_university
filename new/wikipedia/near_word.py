#似た単語を出力するスクリプト
#対象「じゃない」教師データを集める際に活用する

from gensim.models import KeyedVectors

model_data = 'wikipedia.model.bin'

def near_word(keyword):
    #単語の受け皿
    words_array = []
    model = KeyedVectors.load_word2vec_format('./wikipedia/data/' + model_data, binary=True)
    similar_word = model.most_similar(positive=[keyword])
    for i in similar_word[5:]:
        word = i[0].replace('[','').replace(']','')
        words_array.append(word)
    print(words_array)

    return words_array