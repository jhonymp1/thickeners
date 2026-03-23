
# Thicken-app (Streamlit)

Aplicación para análisis de datos de espesamiento (bilingüe ES/EN). Incluye:
- Limpieza automática de nombres
- Remuestreo temporal
- Histogramas con descarga
- Mapa de calor Pearson con descarga
- Correlación móvil
- Modelos: OLS, Random Forest y Polinómico

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura del repositorio
```
thicken-app/
├─ app.py
├─ requirements.txt
├─ .gitignore
├─ README.md
└─ Datos practicando.xlsx   (opcional)
```

## Despliegue en Streamlit Cloud
1. Sube el repositorio a GitHub.
2. Ve a https://share.streamlit.io
3. Selecciona tu repo y rama main.
4. En *Main file path*, coloca: `app.py`
5. Deploy.
```
