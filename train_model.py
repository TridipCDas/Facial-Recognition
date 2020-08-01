from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

#Load our embedded data
embedded_data=pickle.loads(open("output/embeddings.pickle","rb").read())

#Label encoding
le = LabelEncoder()
labels = le.fit_transform(embedded_data["names"])

#Model training
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(embedded_data["embeddings"], labels)

#saving the recognizer on disk
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()