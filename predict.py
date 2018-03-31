import codecs
from bs4 import BeautifulSoup
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import model_from_json
import numpy as np
import random, sys

fp = codecs.open("./output_txt.txt", "r", encoding="utf-8")
#print(fp.read())

#sys.exit(0)
text = fp.read()
print('코퍼스의 길이: ', len(text))
# 문자를 하나하나 읽어 들이고 ID 붙이기
chars = sorted(list(set(text)))
print('사용되고 있는 문자의 수:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars)) # 문자 → ID
indices_char = dict((i, c) for i, c in enumerate(chars)) # ID → 문자
# 텍스트를 maxlen개의 문자로 자르고 다음에 오는 문자 등록하기
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('학습할 구문의 수:', len(sentences))
print('텍스트를 ID 벡터로 변환합니다...')



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

input_txt = "산업통상자원부은 유관기관의 역량을 결집하여 중견기업 성장을 지원하기 위해 28일(수) ‘중견기업 유관기관 협의회’를 첫 개최해 기관별 중견기업 지원 현황을 점검하고 개선방안을 논의했다.회의는 산업통상자원부 이동욱 중견기업정책관을 비롯해 한국산업기술진흥원 한국중견기업연합회 한국무역보험공사 KDB산업은행 대한무역투자진흥공사(KOTRA) 등 10개 중견기업 유관기관과 전문가가 참석하여 지난 2월 발표한 「중견기업 비전 2280」 후속조치 이행상황을 점검하고 기관간 협업 활성화 지원사업간 연계 강화 방안 등을 논의했다.이번 논의에 따라 중견기업연합회은 중견기업 ‘일자리 드림 페스티벌’ 개최 중견기업이 청년을 직접 찾아가는 ‘중견기업 캠퍼스 스카우트(연5회 예정)’ 등을 통해 우수 청년인재가 유망 중견기업에 취업할 수 있는 기회를 확대하기로 했다."
X = input_txt[:20]
Y = "‘중견–중소·벤처기업’ 상생혁신 R&D ’19년 신규 추진"

generated = ''
generated += X

for i in range(400):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(X):
        x[0, t, char_indices[char]] = 1.
    # 다음에 올 문자를 예측하기
    preds = loaded_model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    # 출력하기
    generated += next_char
    X = X[1:] + next_char
    sys.stdout.write(next_char)
    sys.stdout.flush()
print()

# evaluate loaded model on test data
"""
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
"""