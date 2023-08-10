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
import cv2
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
warnings.filterwarnings("ignore")

IMAGEDIR = "images/"

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
# Loading Ear Infection Model here.
earinfection_classifier = load_model('./models/earinfection.h5')
# Loading Tooth Infection Model here.
toothinfection_classifier = load_model('./models/toothinfection.h5')
# Loading Flea-Infection Model here.
pickle_in = open("./models/fleas_ml_model.pkl", "rb")
fleasinfection_ml_classifier = pickle.load(pickle_in)
fleasinfection_cnn_classifier = load_model('./models/fleas_cnn_model.h5')
# Loading Constipation Model here.
pickle_in = open("./models/constipation_ml.pkl", "rb")
constipation_ml_classifier = pickle.load(pickle_in)
constipation_cnn_classifier = load_model('./models/constipation_cnn.h5')

@app.get('/')
def main():
    return {'message': 'Call the `/predict/<disease>` to start the prediction with the data.!'}

# Diarrhea API running at "/predict/diarrhea/"
@app.post('/predict/diarrhea')
def predict(data : diarrhea_class):
    age             = int(data.age)
    blood_presence  = int(data.blood_presence)
    consistency     = int(data.consistency)
    diet_changes    = int(data.diet_changes)
    breed           = int(data.breed)
    prediction      = int(diarrhea_classifier.predict([[age, blood_presence, consistency, diet_changes, breed]]))
    diarrhea_data   = open("user_data/diarrhea.txt", "a")
    new_row         = str(age) + ", " + str(blood_presence) + ", " + str(data.consistency) + ", " +  \
                        str(data.diet_changes) + ", " + str(data.breed) + ", " + str(prediction) + ", NA \n" 
    print(new_row)
    diarrhea_data.write('\n' + (new_row))
    diarrhea_data.close()
    return {"prediction": prediction}

# Jaundice API running at "/predict/jaundice/"
@app.post('/predict/jaundice')
def predict(data : jaundice_class):
    vomiting                    = int(data.vomiting)  
    diarrhoea                   = int(data.diarrhoea)  
    lethargy                    = int(data.lethargy)  
    fever                       = float(data.fever)
    abdominal_pain              = int(data.abdominal_pain)  
    loss_of_appetite            = int(data.loss_of_appetite)  
    paleness                    = int(data.paleness)  
    yellowish_skin              = int(data.yellowish_skin)  
    change_in_urine_feces       = int(data.change_in_urine_feces)  
    polyuria                    = int(data.polyuria)  
    polydipsia                  = int(data.polydipsia)  
    mental_confusion            = int(data.mental_confusion)  
    weight_loss                 = float(data.weight_loss)
    bleeding                    = int(data.bleeding)  
    prediction                  = int(jaundice_classifier.predict([[vomiting, diarrhoea, lethargy, fever, abdominal_pain, loss_of_appetite, paleness, yellowish_skin, change_in_urine_feces, polyuria, polydipsia, mental_confusion, weight_loss, bleeding]]))
    jaundice_data               = open("user_data/jaundice.txt", "a")
    new_row                     = str(vomiting) + ", " + str(diarrhoea) + ", " + str(lethargy) + ", " + str(fever) + ", " + \
                                    str(abdominal_pain) + ", " +  str(loss_of_appetite) + ", " + str(paleness) + ", " + \
                                    str(yellowish_skin) + ", " +  str(change_in_urine_feces) + ", " + str(polyuria) + ", " + \
                                    str(polydipsia) + ", " + str(mental_confusion) + ", " + str(weight_loss) + ", " + str(bleeding) + ", " + \
                                    str(prediction) + ", NA \n" 
    print(new_row)
    jaundice_data.write('\n' + (new_row))
    jaundice_data.close()
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
    obesity_data = open("user_data/obesity.txt", "a")
    new_row = str(file.filename) + ", " + str(cnn_prediction) + ", NA \n" 
    print(new_row)
    obesity_data.write('\n' + (new_row))
    obesity_data.close()
    return {"prediction": cnn_prediction}
    
# Ear Infection API running at "predict/earinfection/"
@app.post("/predict/earinfection")
async def create_upload_file(file: UploadFile = File(...)):
    # Predicting from CNN.
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"images/earinfection/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./images/earinfection/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = earinfection_classifier.predict(np.expand_dims(resize/255, 0))[0]
    print("CNN Prediction: ", cnn_prediction)
    cnn_prediction = str(cnn_prediction)
    earinfection_data = open("user_data/earinfection.txt", "a")
    new_row = str(file.filename) + ", " + str(cnn_prediction) + ", NA \n" 
    print(new_row)
    earinfection_data.write('\n' + (new_row))
    earinfection_data.close()
    return {"prediction": cnn_prediction}

# Ear Infection API running at "predict/earinfection/"
@app.post("/predict/toothinfection")
async def create_upload_file(file: UploadFile = File(...)):
    # Predicting from CNN.
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"images/toothinfection/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./images/toothinfection/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = toothinfection_classifier.predict(np.expand_dims(resize/255, 0))[0]
    print("CNN Prediction: ", cnn_prediction)
    cnn_prediction = str(cnn_prediction)
    toothinfection_data = open("user_data/toothinfection.txt", "a")
    new_row = str(file.filename) + ", " + str(cnn_prediction) + ", NA \n" 
    print(new_row)
    toothinfection_data.write('\n' + (new_row))
    toothinfection_data.close()
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
    eyeinfection_cnn_data = open("user_data/eyeinfection_cnn.txt", "a")
    new_row = str(file.filename) + ", " + str(cnn_prediction) + ", NA \n" 
    print(new_row)
    eyeinfection_cnn_data.write('\n' + (new_row))
    eyeinfection_cnn_data.close()
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
    eyeinfection_ml_data = open("user_data/eyeinfection_ml.txt", "a")
    new_row = str(age) + ", " + str(breed) + ", " + str(sex) + ", " +  \
                str(redness) + ", " + str(swelling) + ", " + str(discharge) + ", " + str(prediction) + ", NA \n" 
    print(new_row)
    eyeinfection_ml_data.write('\n' + (new_row))
    eyeinfection_ml_data.close()
    return {"prediction": prediction}


# Flea Infection API running at "predict/fleasinfection"
@app.post("/predict/fleasinfection")
async def create_upload_file(data: fleas_infection_data = Depends(), file: UploadFile = File(...)):
    # Predicting from CNN.
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./images/fleasinfection/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./images/fleasinfection/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = fleasinfection_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0]
    fleasinfection_cnn_data = open("user_data/fleasinfection_cnn.txt", "a")
    new_row = str(file.filename) + ", " + str(cnn_prediction) + ", NA \n" 
    print(new_row)
    fleasinfection_cnn_data.write('\n' + (new_row))
    fleasinfection_cnn_data.close()
    print("CNN Prediction: ", cnn_prediction)
    # Predicting from ML Model.
    itchingandscratching = int(data.itchingandscratching)
    hairlossorbaldpatches = int(data.hairlossorbaldpatches)
    redorinflamedskin = int(data.redorinflamedskin)
    fleadirtorfleaeggs = int(data.fleadirtorfleaeggs)
    biteorscratchwounds = int(data.biteorscratchwounds)
    coatlength = int(data.coatlength)
    coattype = int(data.coattype)
    currentseason = int(data.currentseason)
    location = int(data.location)
    ml_prediction = (fleasinfection_ml_classifier.predict([[itchingandscratching, hairlossorbaldpatches, redorinflamedskin, 
                            fleadirtorfleaeggs, biteorscratchwounds, coatlength, coattype, currentseason, location]])[0])
    print("ML Prediction: ", ml_prediction)
    prediction = str(ml_prediction + cnn_prediction)
    fleasinfection_ml_data = open("user_data/fleasinfection_ml.txt", "a")
    new_row = str(itchingandscratching) + ", " + str(hairlossorbaldpatches) + ", " + str(redorinflamedskin) + ", " +  \
                str(fleadirtorfleaeggs) + ", " + str(biteorscratchwounds) + ", " + str(coatlength) + ", " +  \
                str(coattype) + ", " + str(currentseason) + ", " + str(location) + ", " + str(prediction) + ", NA \n" 
    print(new_row)
    fleasinfection_ml_data.write('\n' + (new_row))
    fleasinfection_ml_data.close()
    prediction = str(ml_prediction + cnn_prediction)
    return {"prediction": prediction}


# Constipation API running at "predict/constipation"
@app.post("/predict/constipation")
async def create_upload_file(data: constipation_class = Depends(), file: UploadFile = File(...)):
    # Predicting from CNN.
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./images/constipation/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./images/constipation/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = constipation_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0]
    constipation_cnn_data = open("user_data/constipation_cnn.txt", "a")
    new_row = str(file.filename) + ", " + str(cnn_prediction) + ", NA \n" 
    print(new_row)
    constipation_cnn_data.write('\n' + (new_row))
    constipation_cnn_data.close()
    print("CNN Prediction: ", cnn_prediction)
    # Predicting from ML Model.
    infrequent_or_absent_bowel_movements = int(data.infrequent_or_absent_bowel_movements)
    small_hard_dry_stools = int(data.small_hard_dry_stools)
    visible_discomfort_in_abdomen = int(data.visible_discomfort_in_abdomen)
    lack_of_appetite = int(data.lack_of_appetite)
    lethargy_or_unusual_behavior = int(data.lethargy_or_unusual_behavior)
    vomiting = int(data.vomiting)
    ml_prediction = (constipation_ml_classifier.predict([[infrequent_or_absent_bowel_movements, small_hard_dry_stools, visible_discomfort_in_abdomen, 
                            lack_of_appetite, lethargy_or_unusual_behavior, vomiting]])[0])
    print("ML Prediction: ", ml_prediction)
    constipation_ml_data = open("user_data/constipation_ml.txt", "a")
    new_row = str(infrequent_or_absent_bowel_movements) + ", " + str(small_hard_dry_stools) + ", " + str(visible_discomfort_in_abdomen) + ", " +  \
                str(lack_of_appetite) + ", " + str(lethargy_or_unusual_behavior) + ", " + str(vomiting) + ", " +  ", NA \n" 
    print(new_row)
    constipation_ml_data.write('\n' + (new_row))
    constipation_ml_data.close()
    prediction = str(ml_prediction + cnn_prediction)
    return {"prediction": prediction}