import streamlit as st
# Ajusta las importaciones asumiendo que app.py se ejecuta desde el directorio raíz del proyecto
from apps.inversor import run as run_inversor
from apps.buscador import run as run_buscador
from apps.welcome import run as run_welcome

def app_main():


    # Personalizar la barra lateral con CSS
    st.markdown(
        """
        <style>
            /* Tu CSS personalizado aquí */
        </style>
        """,
        unsafe_allow_html=True
    )
     # Asegúrate de que la ruta de la imagen es correcta. Utiliza rutas relativas adecuadas.
    st.sidebar.image("assets/idealista.svg", width=200)  # Asegúrate de que la ruta a la imagen es correcta
    st.sidebar.write("Desarrollado con ❤️ para inversores inteligentes")

    # Texto de bienvenida en la barra lateral
    #st.sidebar.markdown("# Selecciona ")

    # Creación del desplegable en la barra lateral
    app_mode = st.sidebar.selectbox("Elige la aplicación:", ["—", "Inversor", "Buscador"])
    st.sidebar.write()

    if app_mode == "Inversor":
        run_inversor()
    elif app_mode=="—":
        run_welcome()
    elif app_mode == "Buscador":
        run_buscador()

    

if __name__ == "__main__":
    app_main()