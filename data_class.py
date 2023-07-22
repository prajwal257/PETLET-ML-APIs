from pydantic import BaseModel
class diarrhea_class(BaseModel):
    age : int
    blood_presence : int
    consistency : int
    diet_changes : int
    breed : int

class jaundice_class(BaseModel):
    vomiting               : int
    diarrhoea              : int
    lethargy               : int
    fever                  : float
    abdominal_pain         : int
    loss_of_appetite       : int
    paleness               : int
    yellowish_skin         : int
    change_in_urine_feces  : int
    polyuria               : int
    polydipsia             : int
    mental_confusion       : int
    weight_loss            : float
    bleeding               : int

class Options (BaseModel):
    FileName: str
    FileDesc: str = "Upload for demonstration"

class eye_infection_class(BaseModel):
    Age : int
    Breed : int
    Sex : int
    Redness : int
    Swelling : int
    Discharge : int

class fleas_infection_data(BaseModel):
    itchingandscratching : int
    hairlossorbaldpatches : int
    redorinflamedskin : int
    fleadirtorfleaeggs : int
    biteorscratchwounds : int
    coatlength : int
    coattype : int
    currentseason : int
    location : int