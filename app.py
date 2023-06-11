from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import FileResponse
import pandas as pd
import pickle
from data_class import *
import tensorflow as tf
from random import randint
import uuid
import numpy as np
import warnings
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
warnings.filterwarnings("ignore")

IMAGEDIR = "images/"

# Importing the user_data files for merging the new user data.
diarrhea_data = pd.read_csv("user_data/diarrhea.csv")
jaundice_data = pd.read_csv("user_data/jaundice.csv")
eyeinfection_cnn_data = pd.read_csv("user_data/eyeinfection_cnn.csv")
eyeinfection_ml_data = pd.read_csv("user_data/eyeinfection_ml.csv")
obesity_data = pd.read_csv("user_data/obesity.csv")

app = FastAPI()

# Loading diarrhea Model here.
pickle_in = open("models/diarrhea.pkl", "rb")
diarrhea_classifier = pickle.load(pickle_in)
# Loading Jaundice Model here.
pickle_in = open("models/jaundice.pkl", "rb")
jaundice_classifier = pickle.load(pickle_in)
# Loading Eye-Infection Model here.
pickle_in = open("models/eyeinfection_ml.pkl", "rb")
eyeinfection_ml_classifier = pickle.load(pickle_in)
eyeinfection_cnn_classifier = load_model('./models/eyeinfection_cnn.h5')
# Loading Obesity Model here.
obesity_classifier = load_model('./models/obesity_classifier.h5')

@app.get('/')
def main():
    return {'message': 'Call the `/predict/<disease>` to start the prediction with the data.!'}

# Diarrhea API running at "/predict/diarrhea/"
@app.post('/predict/diarrhea')
def predict(data : diarrhea_class):
    age = int(data.Age)
    consistency = int(data.Consistency)
    blood_presence = int(data.Blood_presence)
    diet_changes = int(data.Diet_changes)
    prediction = int(diarrhea_classifier.predict([[age, consistency, blood_presence, diet_changes]]))
    diarrhea_data.loc[len(diarrhea_data.index)] = [age, consistency, blood_presence, diet_changes, prediction, "NA"]
    diarrhea_data.to_csv('user_data/diarrhea.csv')
    return {"prediction": prediction}

# Jaundice API running at "/predict/jaundice/"
@app.post('/predict/jaundice')
def predict(data : jaundice_class):
    vomiting              = int(data.vomiting)  
    diarrhoea             = int(data.diarrhoea)  
    lethargy              = int(data.lethargy)  
    fever                 = float(data.fever)
    abdominal_pain        = int(data.abdominal_pain)  
    loss_of_appetite      = int(data.loss_of_appetite)  
    paleness              = int(data.paleness)  
    yellowish_skin        = int(data.yellowish_skin)  
    change_in_urine_feces = int(data.change_in_urine_feces)  
    polyuria              = int(data.polyuria)  
    polydipsia            = int(data.polydipsia)  
    mental_confusion      = int(data.mental_confusion)  
    weight_loss           = float(data.weight_loss)
    bleeding              = int(data.bleeding)  
    prediction = int(jaundice_classifier.predict([[vomiting, diarrhoea, lethargy, fever, abdominal_pain, loss_of_appetite, paleness, yellowish_skin, change_in_urine_feces, polyuria, polydipsia, mental_confusion, weight_loss, bleeding]]))
    # jaundice_data.loc[len(jaundice_data.index)] = [vomiting, diarrhoea, lethargy, fever, abdominal_pain, loss_of_appetite, paleness, yellowish_skin, change_in_urine_feces, polyuria, polydipsia, mental_confusion, weight_loss,bleeding, prediction, "NA"]
    # jaundice_data.to_csv('user_data/jaundice.csv', index=False)
    return {"prediction": prediction}

# Obesity API running at "predict/obesity/"
@app.post("/predict/obesity")
async def create_upload_file(file: UploadFile = File(...)):
    # Predicting from CNN.
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"images/obesity/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./images/obesity/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = obesity_classifier.predict(np.expand_dims(resize/255, 0))[0]
    print("CNN Prediction: ", cnn_prediction)
    cnn_prediction = str(cnn_prediction)
    # making the dataframe from perdiction.
    # obesity_data.loc[len(obesity_data.index)] = [file.filename, cnn_prediction, "NA"]
    # obesity_data.to_csv('user_data/obesity.csv')
    return {"prediction": cnn_prediction}

# Eyeinfection API running at "predict/eyeinfection"
@app.post("/predict/eyeinfection")
async def create_upload_file(data: eye_infection_class = Depends(), file: UploadFile = File(...)):
    # Predicting from CNN.
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./images/eyeinfection/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./images/eyeinfection/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = eyeinfection_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0]
    # eyeinfection_cnn_data.loc[len(eyeinfection_cnn_data.index)] = [file.filename, cnn_prediction, "NA"]
    # eyeinfection_cnn_data.to_csv('eyeinfection_cnn.csv')
    print("CNN Prediction: ", cnn_prediction)
    # Predicting from ML Model.
    age = int(data.Age)
    breed = int(data.Breed)
    sex = int(data.Sex)
    redness = int(data.Redness)
    swelling = int(data.Swelling)
    discharge = int(data.Discharge)
    ml_prediction = (eyeinfection_ml_classifier.predict([[age, breed, sex, redness, swelling, discharge]])[0])
    print("ML Prediction: ", ml_prediction)
    prediction = str(ml_prediction + cnn_prediction)
    # eyeinfection_ml_data.loc[len(eyeinfection_ml_data.index)] = [age, breed, sex, redness, swelling, discharge, prediction, "NA"]
    # eyeinfection_ml_data.to_csv('ml_dataset.csv')
    return {"prediction": prediction}
 