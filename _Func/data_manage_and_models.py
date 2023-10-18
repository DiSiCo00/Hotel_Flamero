import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from transformers import pipeline

import streamlit as st

import nltk
from translate import Translator
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import random
import json

import joblib

use_cols = ['Noches','Tip_Hab_Fra','R_Factura', 'AD', 'NI','CU','Horario_Venta',
            'P_Alojamiento','P_Desayuno', 'P_Almuerzo', 'P_Cena',
            'Cantidad_Habitaciones','Mes_Entrada','Mes_Venta','Antelacion']


def load_cancel_data():
    #Leemos el csv para recuperar el dataframe
    return pd.read_csv('_Data/cancelaciones.csv')

def load_booking_data():
    #Leemos el csv reservas_total_preprocesado para recuperar el dataframe
    reservas_total=pd.read_csv('_Data/reserva_preprocesado.csv')

    # Convertimos las columnas en formato de fecha
    reservas_total['Fecha entrada'] = pd.to_datetime(reservas_total['Fecha entrada'], dayfirst=True, format = "mixed")
    reservas_total['Fecha venta'] = pd.to_datetime(reservas_total['Fecha venta'], dayfirst=True, format = "mixed")
    reservas_total['Fecha Anulacion'] = pd.to_datetime(reservas_total['Fecha Anulacion'], dayfirst=True, format = "mixed")

    return reservas_total


def train_model(data, _Y_use_cols, Reg_Cls_flag = True, _X_use_cols=use_cols ):
    #Definimos las variables que usaremos en el modelo

    #Dividimos en X e y
    _X = data[_X_use_cols]
    _y = data[_Y_use_cols ]
    _X = pd.get_dummies(_X, columns=["Tip_Hab_Fra", "R_Factura","Horario_Venta", "Mes_Entrada", "Mes_Venta"], drop_first=True)
    robust_scaler = RobustScaler()
    _X[["P_Alojamiento", "Antelacion"]] = robust_scaler.fit_transform(_X[["P_Alojamiento", "Antelacion"]])
    # Inicializamos el escalador Min-Max
    scaler = MinMaxScaler()
    # Aplicamos la normalización
    _X = scaler.fit_transform(_X)

    # Dividimos el conjunto normalizado de datos en entrenamiento, prueba y validación
    X_train, X_test, y_train, y_test = train_test_split(_X,_y, test_size=0.2, random_state=42)

    if Reg_Cls_flag:
        model = RandomForestRegressor(max_depth= 19, n_estimators= 50)
    else:
        model = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='sqrt',
                                    bootstrap=True, max_samples=2/3, oob_score=True)
    model.fit(X_train, y_train)

    return model


# Recopilar datos de la nueva reserva:
def new_Booking(df, room_type, noches, adultos, child, cunas, fecha_entrada):
    
    def get_horario():
        hora = int(datetime.now().strftime('%H'))
        if (0 <= hora < 6):
            return'Madrugada'
        elif (6 <= hora < 12):
            return 'Mañana'
        elif (12 <= hora < 18):
            return'Tarde'
        else:
            return 'Noche'
    

    # Calcular la cantidad de ahbitaciones basados en la cantidad de personas
    def habitaciones(adultos,childs,tipo_habitacion):
        cont=0

        #Si es una SUITE, caben 2 adultos y 2 ni�os o 3 adultos
        if tipo_habitacion=='SUITE':
            cont=childs//2
            adultos-=cont*2+childs%2
            cont+=adultos//3+((adultos%3)+1)//2

        if tipo_habitacion == 'DVC':
            cont=adultos//2+adultos%2
            childs-=cont
        if childs>0:
            cont+=childs//3+((childs%3)+1)//2

        if tipo_habitacion=='DVM':
            cont=adultos//2+adultos%2

        if tipo_habitacion=='IND':
            cont=adultos

        if tipo_habitacion == 'A':
            cont=adultos//4+((adultos%4)+2)//3
            childs-=cont+adultos%4
        if childs>0:
            cont+=childs//3+((childs%3)+1)//2

        if tipo_habitacion in ('EC','EM','DSC','DSM'):
            cont+=adultos//3+((adultos%3)+1)//2
            childs-=cont+adultos%3
        if childs>0:
            cont+=childs//2+childs%2

        return cont

    av_regimen = df["R_Factura"].loc[df['Tip_Hab_Fra']== room_type].value_counts(normalize=True)
    regimen = random.choices(av_regimen.index, av_regimen.values, k=1)
    precio_alojamiento=df['P_Alojamiento'].loc[df['Tip_Hab_Fra'] == room_type].mean()
    precio_desayuno=df['P_Desayuno'].loc[df['R_Factura'] == regimen[0]].mean()
    precio_almuerzo=df['P_Almuerzo'].loc[df['R_Factura'] == regimen[0]].mean()
    precio_cena= df['P_Cena'].loc[df['R_Factura'] == regimen[0]].mean()
    
    # precio_total=precio_alojamiento+precio_desayuno+precio_almuerzo+precio_cena






    fecha_reserva = datetime.now()

    obj = {
    "Noches": noches,
    "Tip_Hab_Fra" : room_type,
    "R_Factura": regimen[0],
    "AD": adultos,
    "NI":child,
    "CU":cunas,
    'Horario_Venta': get_horario(),
    'P_Alojamiento': precio_alojamiento,
    'P_Desayuno': precio_desayuno,
    'P_Almuerzo': precio_almuerzo,
    'P_Cena': precio_cena,
    "Cantidad_Habitaciones": habitaciones(adultos,child,room_type),
    'Mes_Entrada' : fecha_entrada.strftime('%B'),
    'Mes_Venta': datetime.now().strftime('%B'),
    'Antelacion': (fecha_entrada-fecha_reserva).days
    }



    return obj


def new_data_to_model(df, _obj, _use_cols = use_cols):
    #Tomamos nuestra base de entrenamiento para realizar el proceso de normalizaci�n y One Hot Encoding
    _sample = df[_use_cols]

# Agregar la nueva fila al DataFrame
    _X =  pd.concat([_sample, pd.DataFrame(_obj,index=[0])], ignore_index=True)

    #One Hot Encoding de las variables categ�ricas
    _X = pd.get_dummies(_X, columns=["Tip_Hab_Fra", "R_Factura","Horario_Venta", "Mes_Entrada", "Mes_Venta"], drop_first=True)

    #Aplicamos el escalador robusto
    robust_scaler = RobustScaler()
    _X[["P_Alojamiento", "Antelacion"]] = robust_scaler.fit_transform(_X[["P_Alojamiento", "Antelacion"]])

    # Aplicamos la normalizaci�n Min Max
    scaler = MinMaxScaler()
    X = scaler.fit_transform(_X)
    return X


#Funci�n para predercir la probabilidad de cancelaci�n de una reserva con un modelo determinado
def predict_cancel_prob(X):
    model = joblib.load("cls_random_forest.pkl")
    return model.predict_proba(X[-1].reshape(1, -1))[0,1]

    #Predecimos la probabilidad de cancelaci�n de la nueva reserva

#Fecha maxima para cancelar
def cancel_date(X, _obj):
    model = joblib.load("reg_random_forest.pkl")

    _score = model.predict(X[-1].reshape(1, -1))
    # pred=predict_model(model,obj)
    _days = float(_score)*_obj["Antelacion"]


    # Obtener la fecha actual del sistema
    fecha_actual = datetime.now()

    # Sumar d�as a la fecha actual
    _cancel_date = fecha_actual + timedelta(_days)
    _cancel_date = _cancel_date.strftime("%d/%m/%Y")

    return _cancel_date, _score[0]

def fix_cuote(_cancel_prob, _score):
    if _cancel_prob <= 0.50:
        return 0
    elif _cancel_prob > 0.75:
        return 0.5
    else:
        return _score*0.5*_cancel_prob



def predictions(room_type, noches, adultos, child, cunas, fecha_entrada):
    cancel_data = load_cancel_data()
    reservas = load_booking_data()

    obj = new_Booking(reservas, room_type, noches, adultos, child, cunas, fecha_entrada)

    X_booking = new_data_to_model(reservas, obj)

    X_cancel = new_data_to_model(cancel_data, obj)

    cancel_prob = predict_cancel_prob(X_booking)
    c_date, score = cancel_date(X_cancel, obj)

    cuota = fix_cuote(cancel_prob, score)

    return cancel_prob, c_date, cuota, obj, score

def stentiment_analizis(_text):

    nltk.download('vader_lexicon')
    nltk.download('punkt')

    sia = SentimentIntensityAnalyzer()

    translator = Translator(from_lang="es", to_lang="en")
    text = translator.translate(_text)

    palabras_positivas = ["good","happy","big","recommend","nice"]
    palabras_negativas = ["old","small","uncomfortable","bad","slow"]


    def calcular_puntuacion_sentimiento(frase_ingles):
        tokens = nltk.word_tokenize(frase_ingles)
        puntuacion_sentimiento = 0
        for token in tokens:
            if token in palabras_positivas:
                puntuacion_sentimiento += 1
            elif token in palabras_negativas:
                puntuacion_sentimiento -= 1

        return puntuacion_sentimiento
    
    puntuacion = calcular_puntuacion_sentimiento(text)
    sentimiento = sia.polarity_scores(text)

    return sentimiento['compound']


# def cat_raiting(_text, score):
#     categorias = ["Limpieza", "Confort", "Ubicación", "Instalaciones", "Personal"]

#     classifier = pipeline("zero-shot-classification")
#                         # model="facebook/bart-large-mnli",
#                         # revision = "c626438")
#     joblib.dump(classifier, "cero_shut_classifier.pkl")

#     resultados = classifier(_text, categorias)

#     with open("_Data/obj.json") as file:
#         obj = json.load(file)
#         file.close()
#     obj[resultados['labels'][0]]["len"] = obj[resultados['labels'][0]]["len"]+1
#     obj[resultados['labels'][0]]["Score"] = (obj[resultados['labels'][0]]["Score"]*obj[resultados['labels'][0]]["len"] + score)/obj[resultados['labels'][0]]["len"]

#     obj['General']["len"] = obj['General']["len"] + 1
#     obj['General']["Score"] = (obj['General']["Score"] * obj['General']["len"] + score)/obj['General']["len"]

def update_comments_data(_obj):
    df_comments = pd.concat([pd.DataFrame(_obj,index=[0]), pd.read_csv("_Data/comments.csv")])
    df_comments.to_csv("_Data/comments.csv", index=False)