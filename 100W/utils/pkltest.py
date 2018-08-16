import pickle


with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

print(type(word2idx))
print(len(word2idx))
# print(word2idx)
