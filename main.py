import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu


from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import pandas as pd
import transformers


import json
import time

import joblib

# joblib.load("cls_random_forest.pkl")
# joblib.load("reg_random_forest.pkl")

from _Func.html_func import html_sheader
from _Func.html_func import html_score_badges
from _Func.html_func import comment_section

from _Func.data_manage_and_models import load_cancel_data
from _Func.data_manage_and_models import load_booking_data
from _Func.data_manage_and_models import new_Booking
from _Func.data_manage_and_models import update_comments_data
# from _Func.data_manage_and_models import cat_raiting
from _Func.data_manage_and_models import stentiment_analizis
from _Func.data_manage_and_models import predictions

with open("_Data/room_type.json", encoding='utf-8') as file:
    room_type_obj = json.load(file)
    file.close()


st.set_page_config(layout= "wide",
                    page_title = "FlameroHotel")


def add_style(css_file):
    with open(css_file) as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

add_style("_CSS/main.css")

with open("_Data/obj.json") as file:
    obj = json.load(file)
    file.close()

c_main = st.container()

c_body = c_main.container()


with st.sidebar:
    page_selected = option_menu(
    menu_title="Menu",
    options=["Flamero", "Opiniones", "Contacto"],
    default_index=0,
    )


if page_selected == "Flamero":
    
    img =  Image.open("Images/1.png")
    c_body.image(img, use_column_width = "always" )
    c_body.divider()


    with st.form("booking_info"):
        c_body.markdown('<h3>Compruebe disponibilidad:</h3>', unsafe_allow_html=True)

        today = pd.to_datetime(c_body.date_input(label = "¿Qué día es hoy?",
                min_value=pd.to_datetime(datetime.now()),
                max_value=pd.to_datetime('30/9/2024',dayfirst=True),
                on_change=None), dayfirst=True)
        
        entry_date = pd.to_datetime(c_body.date_input(label = "Seleccione la fecha de entrada (Las fechas están acotadas para los días disponibles):",
                value = pd.to_datetime('15/6/2024', dayfirst=True),
                min_value=pd.to_datetime('15/6/2024', dayfirst=True),
                max_value=pd.to_datetime('30/9/2024',dayfirst=True),
                on_change=None, format="DD/MM/YYYY"), dayfirst=True)
            
        col_1, col_2, col_3 = c_body.columns(3)

        noches = int(col_1.number_input('Seleccione la cantidad de noches:',min_value=1))

        adultos = int(col_2.number_input('Cantidad de adultos:',min_value=1))

        child = int(col_3.number_input('Cantidad de niños:',min_value=0))

        cunas=int(col_3.number_input('Seleccione el número de cunas:',min_value=0))

        if child>0:
            room_type_id_pointer = col_1.radio('Seleccione un tipo de habitacion que desea:',
                            ['INDIVIDUAL','ESTUDIO COTO','ESTUDIO MAR','DOBLE SUPERIOR COTO', 'DOBLE SUPERIOR MAR',
                            'APARTAMENTO PREMIUM','DELUXE VISTA COTO', 'DELUXE VISTA COTO', 'SUITE'])
        else:
            room_type_id_pointer = col_1.radio('Seleccione un tipo de habitacion que desea:',
                            ['DOBLE SUPERIOR COTO', 'DOBLE SUPERIOR MAR', 'DELUXE VISTA COTO', 'ESTUDIO COTO', 'ESTUDIO MAR', 'SUITE'])
        
        room_type = room_type_obj[room_type_id_pointer]["ID"]
        
        submitted = st.form_submit_button("Submit")

        c_body.divider()

        
        if submitted:
            with st.spinner("Espera..."):
            
                msg = st.toast('"Recopilando Información"...')
                time.sleep(2)
                # cancel_data = load_cancel_data()
                # reservas = load_booking_data()

                msg.toast("Chequeando disponibilidad...")
                time.sleep(2)
                # obj = new_Booking(reservas, room_type, noches, adultos, child, cunas, entry_date)

                # X_booking = new_data_to_model(reservas, obj)

                # X_cancel = new_data_to_model(cancel_data, obj)

                msg.toast("Chequeando disponibilidad...")
                time.sleep(2)

                # cancel_prob = predict_cancel_prob(X_booking)
                # cancel_date, score = cancel_date(X_cancel, obj)

                msg.toast("Estas de suerte!! Ahora buscaremos lahabitacione adecuada...")

                # cuota = fix_cuote(cancel_prob, score)
                time.sleep(2)
                cancel_prob, c_date, cuota, obj, score = predictions(room_type, noches, adultos, child, cunas, entry_date )

                st.success("Tenemos la habitación adecuada para ti", icon="✅")
            c_room_info = st.expander("Ver Habitación")
            with c_room_info:
                desc_col, info_col = c_room_info.columns(2)
                # Cloumna de Datos
                info_col.markdown(f"<h2>{room_type_id_pointer}:</h2>", unsafe_allow_html=True)
                info_col.divider()
                info_col.markdown(f"<h3>Precio de la estancia:</h3> €{round(obj['P_Alojamiento'], 2)}", unsafe_allow_html=True)
                info_col.markdown(f"<h3>Probabilidad de Cancelación:</h3> {round(cancel_prob*100, 2)}%", unsafe_allow_html=True)
                info_col.markdown(f"<h3>Pago adelantado:</h3> €{round(cuota*obj['P_Alojamiento'] , 2)}", unsafe_allow_html=True)
                info_col.markdown(f"<h3>Cancelación Gratuita Hasta:</h3> {c_date}", unsafe_allow_html=True)
                info_col.markdown(f"<h3>Cancel_Score:</h3> {round(score, 2)}", unsafe_allow_html=True)

                # Columna de Desceipcion de la Habitación
                room_img =  Image.open(f"{room_type_obj[room_type_id_pointer]['img_path']}")
                desc_col.image(room_img, use_column_width="always")
                desc_col.markdown(f"<h6>{room_type_obj[room_type_id_pointer]['Desc']}</h6>", unsafe_allow_html=True)
        


elif page_selected == "Opiniones":
    with c_body:
        img2 =  Image.open("Images/2.png")
        st.image(img2, use_column_width = "always" )
        c_body.divider()
        col_raitings, comments_section = c_body.columns((1,2), gap="small" )
        col_raitings.markdown("<h2>Raitings:</h2>", unsafe_allow_html=True)
        for key, value in list(obj.items()):
            c_raitings = col_raitings.container()
            list , badge = c_raitings.columns(2)
            list.markdown(f"<h4>{key}</h4>", unsafe_allow_html=True)
            badge.markdown(html_score_badges(value["Score"]), unsafe_allow_html=True)
            c_raitings.divider()


        with comments_section:
            comments_section.markdown(html_sheader("Comentarios"), unsafe_allow_html=True)
            comments_section.markdown(f"<div class='comment_section'>{comment_section()}</div>", unsafe_allow_html=True)
            comments_section.divider()
            with comments_section.form(key="Comment_section_form"):
                
                st.subheader("**Quieres compartinos tu experiencia?**")
                user = st.text_input("Escribe tu nombre o un alias con el que desees dejar tu comentario:",
                                    value = "Anónimo")
                text_comment = st.text_area(label="Escribe tu comentario aqui:")
                raiting = int(st.number_input("Califica tu experienia con nosotros entre 1 - 10",
                                        value=10,
                                        placeholder="Type a number...",
                                        max_value=10,
                                        min_value=0))
                submit_com = st.form_submit_button("Enviar")
    if submit_com and text_comment != "":
        # cat_raiting(text_comment, raiting)
        update_comments_data({"Score": raiting,
                            "Comentario_Positivo":text_comment,
                            "Usuario":user})
        if stentiment_analizis(text_comment) >= 0.05:
            c_main.balloons()
            c_main.success("Gracias por su comentario")
            time.sleep(5)
        elif stentiment_analizis(text_comment) <= 0.05:
            c_main.info("Agradecemos que hayas compartido tus preocupaciones con nosotros. Lamentamos mucho que hayas tenido esta experiencia", icon="ℹ️")
            time.sleep(5)
        else:
            c_main.success("Gracias por su comentario")
            time.sleep(5)
        st.rerun()

