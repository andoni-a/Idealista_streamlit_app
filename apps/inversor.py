import streamlit as st
import pydeck as pdk
import pandas as pd
import pickle
from xgboost import XGBRegressor
import requests
import numpy as np


def run():
    st.title("Portal Inversor")
    st.sidebar.header("Dirección")
    calle_buscada = st.sidebar.text_input('Nombre de la calle:', '')
    numero_buscado = st.sidebar.text_input('Número', '')
    st.sidebar.header("Características")
    #precio_oferta = st.sidebar.slider("Precio de Oferta:", 0, 8000000)
    metros_cuadrados = st.sidebar.slider("Metros cuadrados:", 0, 900)
    habitaciones=st.sidebar.selectbox("Habitaciones", [0,1,2,3,4,5,6,7])
    baños=st.sidebar.selectbox("Baños", [0,1,2,3,4])
    estado_vivienda = st.sidebar.selectbox("Estado de la vivienda:", ["Sin reforma", "Con reforma parcial", "Reformado"])
    st.sidebar.header("Reforma")
    inversion = st.sidebar.slider("Inversión en reforma", 0, 100000)
    
    boton_calcular = st.sidebar.button('Calcular')
    
    if boton_calcular:
        #CARGAMOS CALLEJERO
        callejero = pd.read_csv('data/callejero.csv', encoding='utf-8')
        #PREDECIMOS PRECIO
        def predictor(m2,hab,baños,estado,long,lat,distrito_id,barrio_id):
            modelo_url = "https://github.com/anamartiiins/MDS6-IDEALISTA/raw/main/src/modelization/modelo_xgb_try.pickle"
            response = requests.get(modelo_url)
            with open('modelo_xgb_try.pickle', 'wb') as f:
                f.write(response.content)

            # Cargar el modelo desde el archivo pickle
            with open('modelo_xgb_try.pickle', 'rb') as f:
                model_xgb_loaded = pickle.load(f)
            # Obtener las características del modelo
            features_modelo = model_xgb_loaded.get_booster().feature_names
            
            # Crear DataFrame de entrada con algunas columnas
            input_data = pd.DataFrame({
                'area_construida': [m2],
                'n_habitaciones': [hab],
                'n_banos': [baños],
                'buen_estado': [estado],
                'longitud': [long],
                'latitud': [lat],
                'DISTRITO_Arganzuela': [distrito_id],  # Ejemplo de una columna existente
                'BARRIO_Valdezarza': [barrio_id],    # Ejemplo de una columna existente
                })
            # Verificar y agregar las características que faltan
            for feature in features_modelo:
                if feature not in input_data.columns:
                    # Si la característica falta, agrégala con un valor predeterminado (por ejemplo, 0)
                    input_data[feature] = 0
            
            # Organizar el DataFrame en el mismo orden que las características del modelo
            input_data = input_data[features_modelo]
            
            # Hacer la predicción
            prediction = model_xgb_loaded.predict(input_data)
            return prediction[0]
        
        def generar_barrio_id(distrito, barrio):
            distrito_str = f"{int(distrito):02d}"
            barrio_str = f"{int(barrio):03d}"
            return f"0-EU-ES-28-07-001-079-{distrito_str}-{barrio_str}"

        def buscar_calle_numero(calle_buscada, numero_buscado, callejero):
            callejero['VIA_NOMBRE'] = callejero['VIA_NOMBRE'].str.lower()
            callejero['VIA_NOMBRE_ACENTOS'] = callejero['VIA_NOMBRE_ACENTOS'].str.lower()
            calle_buscada = calle_buscada.lower()

            resultado = callejero[(callejero['VIA_NOMBRE'] == calle_buscada) | 
                            (callejero['VIA_NOMBRE_ACENTOS'] == calle_buscada)]

            if numero_buscado.strip():
                try:
                    numero_buscado = int(numero_buscado)
                    resultado = resultado[resultado['NUMERO'] == numero_buscado]
                except ValueError:
                    st.warning("Número inválido. Mostrando resultados para la calle sin especificar número.")

            return resultado

        def estado (estado_vivienda):
            if estado_vivienda=="Sin reforma":
                return 0
            else: 
                return 1

        def calcular_rentabilidad_lineal(distrito, metros_cuadrados, inversion):
            # Obtener los coeficientes de rentabilidad para el distrito específico
            tabla_rentabilidades=pd.read_csv("data/Tabla_Rent_Ref.csv", delimiter=";")
            coeficientes = tabla_rentabilidades.loc[tabla_rentabilidades['Distrito'] == distrito]
            
            # Calcular el costo por metro cuadrado de la reforma
            costo_por_m2 = inversion / metros_cuadrados
            
            # Definir los puntos para la interpolación lineal
            # Asumimos que la rentabilidad aumenta linealmente dentro de cada rango
            puntos = [(0, 1), (250, coeficientes['SR_RP'].values[0]), (500, coeficientes['SR_RC'].values[0])]
            
            # Interpolación lineal
            if costo_por_m2 <= 250:
                rentabilidad = np.interp(costo_por_m2, [p[0] for p in puntos[:2]], [p[1] for p in puntos[:2]])
            elif 250 < costo_por_m2 <= 500:
                rentabilidad = np.interp(costo_por_m2, [p[0] for p in puntos[1:]], [p[1] for p in puntos[1:]])
            else:
                rentabilidad = coeficientes['SR_RC'].values[0]  # Para inversiones > 500€/m² usamos el coeficiente directamente
            
            return ((rentabilidad-1))
        
        if calle_buscada:
            resultado_busqueda = buscar_calle_numero(calle_buscada, numero_buscado, callejero)
            
            if not resultado_busqueda.empty:
                latitud = resultado_busqueda.iloc[0]['LATITUD']
                longitud = resultado_busqueda.iloc[0]['LONGITUD']
                distrito = resultado_busqueda.iloc[0]['DISTRITO_ID']
                barrio = int(resultado_busqueda.iloc[0]['BARRIO_ID'])
                barrio_id = generar_barrio_id(distrito, barrio)
                
                estado_bin=estado(estado_vivienda)
                precio_predict=predictor(metros_cuadrados,habitaciones,baños,estado_bin,longitud,latitud,distrito,barrio)
                precio_compra = precio_predict
                rentabilidad=calcular_rentabilidad_lineal(distrito, metros_cuadrados, inversion)
                
                precio_venta=(precio_compra + inversion)*(1+rentabilidad)

                                                          
               
                url = 'https://raw.githubusercontent.com/anamartiiins/MDS6-IDEALISTA/main/output_data/df_train_util.csv'
                df = pd.read_csv(url, encoding='utf-8')
                barrio_seleccionado = df[df['barrio_id'] == barrio_id]
                barrio_nombre=barrio_seleccionado['barrio'].unique()
                # Mostrar datos de barrio_seleccionado si están disponibles
                if not barrio_seleccionado.empty:
                    #st.dataframe(barrio_seleccionado)
                
                    df_filtrado = barrio_seleccionado.copy()

                # Añadir punto de búsqueda específico al df_filtrado
                punto_especifico = pd.DataFrame({
                    'latitud': [latitud],
                    'longitud': [longitud],
                    'precio': [precio_predict],  # Precio fijo, no se usará para calcular el color
                    'color': [(190, 37, 150, 140)],  # Color específico en formato RGBA
                    'altura': [100]  # Altura fija para destacarlo
                })
                
                precio_venta_formato = "{:,.0f} €".format(precio_venta).replace(",", "X").replace(".", ",").replace("X", ".")
                precio_compra_formato="{:,.0f} €".format((precio_compra+inversion)).replace(",", "X").replace(".", ",").replace("X", ".")
                precio_medio_barrio=df_filtrado['precio'].mean()
                precio_medio_barrio_formato="{:,.0f} €".format(precio_medio_barrio).replace(",", "X").replace(".", ",").replace("X", ".")
                factor_barrio=((precio_medio_barrio/precio_venta)-1)*100
                rentabilidad_formato=rentabilidad*100

                col1, col2 ,col3 = st.columns(3)
                col1.metric("Precio Venta ", precio_venta_formato, f"{rentabilidad_formato:.2f}% rentabilidad")
                col2.metric("Coste Inversion ", precio_compra_formato)
                col3.metric("Precio Medio Barrio", precio_medio_barrio_formato)   

            # Calcular el centro del mapa basado en datos filtrados
            if not df_filtrado.empty:
                centro_lat = df_filtrado['latitud'].mean()
                centro_lon = df_filtrado['longitud'].mean()
            else:
                centro_lat, centro_lon = 40.416775, -3.703790  # Coordenadas del centro de Madrid

            # Calcular el precio mínimo y máximo para la escala de colores
            min_precio = df_filtrado['precio'].min()
            max_precio = df_filtrado['precio'].max()

            # Función para calcular el color basado en el precio
            def calcular_color_y_altura(precio, min_precio, max_precio):
                norm_precio = (precio - min_precio) / (max_precio - min_precio) if max_precio > min_precio else 0
                altura = 10 + norm_precio * (490)  # Altura normalizada entre 10 y 500
                r = int((1 - norm_precio) * 222+150)
                g = int((1 - norm_precio) * 250+190)
                b = int((1 - norm_precio) * 69+37)
                return [r, g, b], altura  # RGBA, 140 es la opacidad

            # Aplicar la función de color y altura a cada fila
            df_filtrado[['color', 'altura']] = df_filtrado.apply(
                lambda x: calcular_color_y_altura(x['precio'], min_precio, max_precio), 
                axis=1, result_type='expand'
            )
            # Capa para punto_especifico (ajustada según tu último requerimiento)
            capa_punto_especifico = pdk.Layer(
                "ColumnLayer",
                data=punto_especifico,
                get_position='[longitud, latitud]',
                get_elevation=10 + ((precio_venta - min_precio) / (max_precio - min_precio)) * (490),
                get_fill_color=[246, 51, 102],
                radius=20,
                pickable=True,
                elevation_scale=1,
                elevation_range=[0, 500],
                auto_highlight=True,
            )
            # Capa para df_filtrado
            capa_df_filtrado = pdk.Layer(
                'ColumnLayer',
                data=df_filtrado,
                get_position='[longitud, latitud]',
                get_elevation='altura',
                get_fill_color='color',
                get_line_color=[150, 190, 37],
                radius=8,
                pickable=True,
                elevation_scale=1,
                elevation_range=[0, 500],
                auto_highlight=True,
                coverage=1,
            )


            # Ajusta la visualización en pydeck
            mapa = pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state={
                    "latitude": latitud if df_filtrado.empty else df_filtrado['latitud'].mean(),
                    "longitude": longitud if df_filtrado.empty else df_filtrado['longitud'].mean(),
                    "zoom": 14,
                    "pitch": 0,
                },
                layers=[capa_df_filtrado, capa_punto_especifico],  # Aquí se combinan ambas capas
            )

        st.pydeck_chart(mapa)
    
if __name__ == "__main__":
    run()