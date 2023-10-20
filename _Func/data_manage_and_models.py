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
def new_Booking(df, room_type, regimen, noches, adultos, child, cunas, today, fecha_entrada):
    
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
    

   #Función para definir la cantidad mínima de habitaciones a reservar en base a huespedes y tipo de habitación
    def habitaciones(adultos, niños, tipo_habitacion):
      cont = 1

      #Si es una SUITE, la capacidad máxima es de 2 adultos y 2 niños o 3 adultos
      if tipo_habitacion == 'SUITE':
        #Si hay más de 2 niños por adulto devolvemos error (0)
        if adultos * 2 < niños:
          return 0

        #Asignamos los niños de 2 en 2 y dos adultos por habitación
        cont = niños // 2 + niños % 2
        adultos -= cont * 2

        #Asignamos habitaciones de 3 adultos
        if  adultos > 0:
          cont += adultos // 3
          adultos = adultos % 3

          #Última habitación si sobran adultos
          if adultos > 0:
            cont += 1

      #Si es una habitación DELUXE VISTA COTO, la capacidad máxima es de 2 adultos y 1 niño
      if tipo_habitacion == 'DVC':
        #Si hay más niños que adultos devolvemos error (0)
        if adultos < niños:
          return 0

        #Asignamos una habitación por niño y 2 adultos por habitación
        cont = niños
        adultos -= cont * 2

        #Asignamos habitaciones de 2 adultos
        if  adultos > 0:
          cont += adultos // 2 + adultos % 2

      #Si es una habitación DELUXE VISTA MAR, la capacidad máxima es de 2 adultos. No se permiten niños
      if tipo_habitacion == 'DVM':
        #Asignamos habitaciones de 2 adultos
        cont = adultos // 2 + adultos % 2

      #Si es una habitación INDIVIDUAL, la capacidad máxima es de 1 adulto. No se permiten niños
      if tipo_habitacion == 'IND':
        #Asignamos las habitaciones individuales
        cont = adultos

      #Si es un APARTAMENTO PREMIUM, la capacidad máxima es de 4 adultos y 3 niños
      if tipo_habitacion == 'A':
        #Si hay más de 3 niños por adulto devolvemos error (0)
        if adultos * 3 < niños:
          return 0

        #Asignamos los niños de 3 en 3 y cuatro adultos por habitación
        cont = niños // 3
        niños = niños % 3
        adultos -= cont * 4

        #Si sobran niños asignamos otra habitación con capacidad para 4 adultos más
        if niños > 0:
          cont += 1
          adultos -= 4

        #Si sobran adultos asignamos habitaciones de 4 adultos
        if adultos > 0:
          cont += adultos // 4
          adultos = adultos % 4

          #Última habitación si sobran adultos
          if adultos > 0:
            cont += 1

      #Si es un ESTUDIO estándar o una habitación DOBLE SUPERIOR, independientemente de si es vista COTO o MAR,
      #la capacidad máxima es de 3 adultos y 1 niño o 2 adultos y 2 niños
      if tipo_habitacion in ('EC', 'EM', 'DSC', 'DSM'):
        #Si hay más de 2 niños por adulto devolvemos error (0)
        if adultos * 2 < niños:
          return 0

        #Asignamos los niños de 2 en 2 y dos adultos por habitación
        cont = niños // 2
        adultos -= cont * 2

        #Asignamos habitaciones de 3 en 3
        if adultos > 0:
          cont += adultos // 3
          adultos = adultos % 3

          #Última habitación si sobran adultos
          if adultos > 0:
            cont += 1
        #Si no sobran adultos pero sí un niño, asignaremos una habitación extra
        elif niños % 2 == 1:
          cont += 1

      return cont

    precio_alojamiento=df['P_Alojamiento'].loc[df['Tip_Hab_Fra'] == room_type].mean()
    precio_desayuno=df['P_Desayuno'].loc[df['R_Factura'] == regimen[0]].mean()
    precio_almuerzo=df['P_Almuerzo'].loc[df['R_Factura'] == regimen[0]].mean()
    precio_cena= df['P_Cena'].loc[df['R_Factura'] == regimen[0]].mean()

    obj = {
    "Noches": noches,
    "Tip_Hab_Fra" : room_type,
    "R_Factura": regimen,
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
    'Mes_Venta': today.strftime('%B'),
    'Antelacion': (fecha_entrada-fecha_today).days
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


#Funci�n para predecir la probabilidad de cancelaci�n de una reserva con un modelo determinado
def predict_prob(X):
    model = joblib.load("cls_compress_random_forest.pkl")
    return model.predict_proba(X[-1].reshape(1, -1))[0,1]

    #Predecimos la probabilidad de cancelaci�n de la nueva reserva

#Fecha maxima para cancelar
def predict_date_score(X, _obj):
    model = joblib.load("reg_random_forest.pkl")

    _score = model.predict(X[-1].reshape(1, -1))
    return _score[0]

def fix_cuote(_cancel_prob, _score):
    if _cancel_prob <= 0.50:
        return 0
    elif _cancel_prob > 0.75:
        return 0.5
    else:
        return _score*0.5*_cancel_prob

#Función cuota no reembolsable
def func_no_reembolso(_obj, _cuota_media=0.10, _cuota_maxima=0.25, _umbral_inferior=0.25, _umbral_superior=0.4, model=random_forest, model_canc=random_forest_canc):
        #Condiciones de control
        if 0 <= _cuota_maxima <= 1:
          if 0 <= _cuota_media <= 1:
            if 0 <= _umbral_inferior <= 1:
              if 0 <= _umbral_superior <= 1:
                if _umbral_superior >_umbral_inferior:

                  #Predicción de la probabilidad de cancelación
                  _pred = predict_prob(_obj, model)

                  #Según los distintos umbrales y dependiendo del score, las cancelaciones tendrán unas cuotas y fechas de cancelación 
                  if _pred < _umbral_inferior:
                    if predict_date_score(_obj,model_canc)<0.5:
                      st.write(f"¡¡Aviso de posible cancelación tardía!!")
                      st.write(f"Riesgo bajo de cancelación.\nEl huésped podrá cancelar sin coste hasta 7 días antes del {_obj['Fecha entrada']}")
                    else:
                      st.write(f"Riesgo bajo de cancelación.\nEl huésped podrá cancelar sin coste hasta 24 horas antes del {_obj['Fecha entrada']}")
                  elif _pred > _umbral_superior:
                    if predict_date_score(_obj,model_canc)<0.5:
                      st.write(f"¡¡Aviso de posible cancelación tardía!!")
                      st.write(f"Riesgo alto de cancelación.\nEl huésped podrá cancelar perdiendo un {(_cuota_maxima)*100:.1f}% del Precio total hasta 30 días antes del {_obj['Fecha entrada']}")
                    else:
                      st.write(f"Riesgo alto de cancelación.\nEl huésped podrá cancelar perdiendo un {(_cuota_maxima)*100:.1f}% del Precio total hasta 7 días antes del {_obj['Fecha entrada']}")
                  else:
                    if predict_date_score(_obj,model_canc)<0.5:
                      st.write(f"¡¡Aviso de posible cancelación tardía!!")
                      st.write(f"Riesgo moderado de cancelación.\nEl huésped podrá cancelar perdiendo un {(_cuota_media)*100:.1f}% del Precio total hasta 14 días antes del {_obj['Fecha entrada']}")
                    else:
                      st.write(f"Riesgo moderado de cancelación.\nEl huésped podrá cancelar perdiendo un {(_cuota_media)*100:.1f}% del Precio total hasta 48 horas antes del {_obj['Fecha entrada']}")
                else:
                  raise ValueError("El valor de ´umbral_superior´  tiene que ser mayor que ´umbral_inferior´.")
              else:
                raise ValueError("El valor ´umbral_superior´ debe estar entre 0 y 1.")
            else:
              raise ValueError("El valor ´umbral_inferior´ debe estar entre 0 y 1.")
          else:
            raise ValueError("El valor ´cuota_media´ debe estar entre 0 y 1.")
        else:
          raise ValueError("El valor ´cuota_maxima´ debe estar entre 0 y 1.")

def predictions(room_type, noches, adultos, child, cunas, fecha_entrada):
    cancel_data = load_cancel_data()
    reservas = load_booking_data()

    obj = new_Booking(reservas, room_type, noches, adultos, child, cunas, fecha_entrada)

    X_booking = new_data_to_model(reservas, obj)

    X_cancel = new_data_to_model(cancel_data, obj)

    cancel_prob = predict_prob(X_booking)
    c_date, score = predict_date_score(X_cancel, obj)

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
