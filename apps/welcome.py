import streamlit as st
import pydeck as pdk
import pandas as pd
import pickle
from xgboost import XGBRegressor
import requests

def run():
    
    

    # Título principal
    st.title("¡Bienvenidos a la revolución inmobiliaria con Idealista!")

# Introducción
    st.write("""
    Prepárate para embarcarte en una aventura inmobiliaria donde tú tienes el control. Si eres un inversor en busca de maximizar tus retornos o simplemente alguien que ama encontrar las mejores ofertas del mercado, has llegado al lugar perfecto. 🏡

    Con Idealista, el poder de transformar el mercado inmobiliario está al alcance de tu mano. Nuestra aplicación está diseñada para brindarte las herramientas que necesitas para tomar decisiones sabias y oportunas, ya sea que estés dando tus primeros pasos en el mundo de las inversiones o seas un veterano en busca de expandir tu portafolio.
    """)

    st.subheader("¿Qué te gustaría hacer hoy?")
    st.write("""
    - **Aplicación de Inversor**: Sumérgete en los detalles de una oportunidad inmobiliaria y deja que nuestra tecnología prediga el precio de mercado. :chart_with_upwards_trend: Con el añadido de calcular el retorno esperado post-reforma, tomar decisiones nunca ha sido tan sencillo.

    - **Buscador de Oportunidades**: Explora un universo de posibilidades con nuestro buscador de oportunidades activas en Idealista. :mag_right: Filtra, busca y encuentra esas joyas ocultas que están esperando ser descubiertas.

    Preparados, listos... ¡a invertir! :rocket:
    """)

    # Pie de página
    st.markdown("---")


    


    
if __name__ == "__main__":
    run()