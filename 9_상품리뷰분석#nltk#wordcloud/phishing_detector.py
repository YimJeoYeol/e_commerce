# -- coding: utf-8 --

import sys
import io
import pickle as pk

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

if __name__ == '__main__':
    with open("naiveBayes.pickle") as f:
        model = pk.load(f)
    prediction = model.predict([sys.argv[1]])
    predict = ["보이스피싱입니다." if prediction else "보이스피싱이 아닙니다."]
    print(predict)
