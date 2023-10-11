import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
actions=np.array([
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
  ])
no_sequences=16
sequence_length=10
data=[]
label=[]
for action in actions:
    for sequence in range(no_sequences):
        for frame_no in range(sequence_length):
            x=np.load('Mp_Data1/{}/{}/{}.npy'.format(action,sequence,frame_no))
        data.append(x)
        label.append(action)


# data_dict = pickle.load(open('./data.pickle', 'rb'))
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])])

data = np.asarray(data)
labels = np.asarray(label)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                    shuffle=True, stratify=labels)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

df = pd.DataFrame(y_predict, y_test)
print(df.head(10))
score = accuracy_score(y_predict, y_test)
print(score)
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
# print(data_dict['labels'])
# print(data_dict)
