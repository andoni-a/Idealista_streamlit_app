import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np


def calcular_rentabilidad_lineal(distrito, metros_cuadrados, inversion):
    # Obtener los coeficientes de rentabilidad para el distrito específico
    tabla_rentabilidades = pd.read_csv("data/Tabla_Rent_Ref.csv", delimiter=";")
    distrito_n= int(distrito.split("-")[-2])
    coeficientes = tabla_rentabilidades.loc[tabla_rentabilidades['Distrito'] == distrito_n]

    # Calcular el costo por metro cuadrado de la reforma
    costo_por_m2 = inversion / metros_cuadrados

    # Definir los puntos para la interpolación lineal
    puntos = [(0, 1), (250, coeficientes['SR_RP'].values[0]), (600, coeficientes['SR_RC'].values[0])]

    # Interpolación lineal
    if costo_por_m2 <= 250:
        rentabilidad = np.interp(costo_por_m2, [p[0] for p in puntos[:2]], [p[1] for p in puntos[:2]])
    elif 250 < costo_por_m2 <= 600:
        rentabilidad = np.interp(costo_por_m2, [p[0] for p in puntos[1:]], [p[1] for p in puntos[1:]])
    else:
        rentabilidad = coeficientes['SR_RC'].values[0]  # Para inversiones > 600€/m² usamos el coeficiente directamente

    return rentabilidad - 1  # Ajuste basado en la fórmula proporcionada

def run():
    st.title("Buscador de Oportunidades")
    # Cargar los datos
    url = 'https://raw.githubusercontent.com/anamartiiins/MDS6-IDEALISTA/main/output_data/df_train_util.csv'
    df = pd.read_csv(url, encoding='utf-8')
    # Selecciona solo las columnas necesarias
    columns_needed = [
        'precio', 'precio_unitario_m2', 'area_construida', 'n_habitaciones',
        'n_banos', 'ascensor', 'aire_acondicionado', 'amueblado', 'parking',
        'piscina', 'portero', 'jardin',  'barrio',"barrio_id","latitud","longitud"
    ]
    df = df[columns_needed]
    # Filtro por barrio
    barrios_unicos = sorted(df['barrio'].unique())
    barrio_seleccionado = st.sidebar.selectbox("Selecciona un barrio:", options=['Todos'] + list(barrios_unicos))
    inversion_min, inversion_max = st.sidebar.slider("Selecciona rango de inversion total:", 0, 800000, (0, 2000000))
    nivel_reforma = st.sidebar.slider("De la inversion total cuanto destinas a reforma:", 0, inversion_max)
    rentabilidad_min, rentabilidad_max = st.sidebar.slider("Rentabilidad esperada:", 0.0, 30.0, (0.0, 30.0))
    boton_calcular = st.sidebar.button('Calcular')
    
    if boton_calcular:
    
        precio_max=inversion_max-nivel_reforma
        precio_mi=inversion_min-nivel_reforma
        
        # Aplicar filtros al DataFrame
        if barrio_seleccionado != 'Todos':
            df_filtrado = df[df['barrio'] == barrio_seleccionado]
        else:
            df_filtrado = df

        df_filtrado = df_filtrado[
            (df_filtrado['precio'] >= precio_mi) & (df_filtrado['precio'] <= precio_max) #&
            #(df_filtrado['nivel_reforma'] >= nivel_reforma_min) & (df_filtrado['nivel_reforma'] <= nivel_reforma_max) &
            #(df_filtrado['rentabilidad'] >= rentabilidad_min) & (df_filtrado['rentabilidad'] <= rentabilidad_max)
        ]
        
        def calcular_rentabilidad_por_fila(row):
            distrito = row['barrio_id']
            metros_cuadrados = row['area_construida']
            inversion = nivel_reforma  # Esto asume que 'nivel_reforma' es un valor constante para todas las filas
            return calcular_rentabilidad_lineal(distrito, metros_cuadrados, inversion)

        # Aplicar la función a cada fila del DataFrame sin usar lambda
        df_filtrado['rentabilidad'] = df_filtrado.apply(calcular_rentabilidad_por_fila, axis=1)
        df_filtrado1 = df_filtrado.sort_values(by='rentabilidad', ascending=False)

        # Selecciona las columnas necesarias en el orden deseado
        columns_ordered = [
            'rentabilidad', 'precio', 'precio_unitario_m2', 'area_construida', 'n_habitaciones',
            'n_banos', 'ascensor', 'aire_acondicionado', 'amueblado', 'parking',
            'piscina', 'portero', 'jardin',  'barrio'
        ]
        df_filtrado1 = df_filtrado1[columns_ordered]
        df_filtrado1 = df_filtrado1[(df_filtrado1['rentabilidad'] >= (rentabilidad_min/100)) & (df_filtrado['rentabilidad'] <= (rentabilidad_max/100))]
        df_filtrado = df_filtrado[(df_filtrado['rentabilidad'] >= (rentabilidad_min/100)) & (df_filtrado['rentabilidad'] <= (rentabilidad_max/100))]
        # Mostrar la tabla filtrada
        st.write("Resultados filtrados:")
        df_filtrado1=df_filtrado1.reset_index(drop=True)
        st.dataframe(df_filtrado1)
        
        # Calcular el centro del mapa basado en datos filtrados
        if not df_filtrado.empty:
            centro_lat = df_filtrado['latitud'].mean()
            centro_lon = df_filtrado['longitud'].mean()
        else:
            centro_lat, centro_lon = 40.416775, -3.703790  # Coordenadas del centro de Madrid

        # Calcular el precio mínimo y máximo para la escala de colores
        min_precio = df_filtrado['precio_unitario_m2'].min()
        max_precio = df_filtrado['precio_unitario_m2'].max()

        # Función para calcular el color basado en el precio
        def calcular_color_y_altura(precio, min_precio, max_precio):
            # Normalizar el precio entre 0 y 1
            norm_precio = (precio - min_precio) / (max_precio - min_precio)
            altura = 10 + norm_precio * (500 - 10)  # Altura normalizada entre 10 y 100
            # Interpolar el color entre #defa45 (rgb(222, 250, 69)) y negro (rgb(0, 0, 0))
            r = int((1 - norm_precio) * 222+150)
            g = int((1 - norm_precio) * 250+190)
            b = int((1 - norm_precio) * 69+37)
            return [r, g, b], altura  # RGBA, 140 es la opacidad

        # Aplicar la función de color a cada fila
        df_filtrado[['color', 'altura_normalizada']] = df_filtrado.apply(lambda x: calcular_color_y_altura(x['precio_unitario_m2'], min_precio, max_precio), axis=1, result_type='expand')


        # Crear un mapa con PyDeck
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=centro_lat,
                longitude=centro_lon,
                zoom=13,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ColumnLayer',
                    data=df_filtrado,
                    get_position='[longitud, latitud]',
                    get_elevation='altura_normalizada',
                    elevation_scale=1,  # La escala de elevación ya está normalizada
                    radius=10,
                    get_fill_color='color',
                    pickable=True,
                    auto_highlight=True,
                ),
            ],
        ))

if __name__ == "__main__":
    run()
