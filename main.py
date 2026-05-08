from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import pickle
import numpy as np
import os

# =========================
# VARIABLES DE ENTORNO
# =========================

BUCKET_NAME = os.getenv("BUCKET_NAME")
MODEL_KEY = os.getenv("MODEL_KEY")

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

REGION = os.getenv("REGION", "us-east-1")

# =========================
# FASTAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONEXIÓN AWS S3
# =========================

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)

# =========================
# CARGAR MODELO
# =========================

def cargar_modelo():

    try:
        print("Cargando modelo desde S3...")

        obj = s3.get_object(
            Bucket=BUCKET_NAME,
            Key=MODEL_KEY
        )

        modelo = pickle.loads(obj['Body'].read())

        print("Modelo cargado correctamente")

        return modelo

    except Exception as e:
        print("Error cargando modelo:", e)
        return None


modelo = cargar_modelo()

# =========================
# MODELO DE ENTRADA
# =========================

class DatosEntrada(BaseModel):
    datos: list

# =========================
# ROOT
# =========================

@app.get("/")
def root():
    return {
        "mensaje": "API funcionando correctamente"
    }

# =========================
# PREDICCIÓN
# =========================

@app.post("/prediccion")
def predecir(data: DatosEntrada):

    try:

        if modelo is None:
            return {"error": "Modelo no cargado"}

        # Convertir input a numpy
        datos = np.array(data.datos).reshape(1, -1)

        # =========================
        # 🔴 REGLA: EXTRAER EDAD
        # =========================
        edad = data.datos[4]  # <- según tu ejemplo

        # 🚫 BLOQUEO MENORES DE 18
        if edad < 18:
            return {
                "probabilidad": 0.0,
                "decision": "RECHAZAR",
                "razon": "Menor de edad"
            }

        print("Datos recibidos:", datos)

        # Predicción del modelo
        prob = modelo.predict_proba(datos)[0][1]

        return {
            "probabilidad": float(prob),
            "decision": (
                "RECHAZAR"
                if prob > 0.5
                else "APROBAR"
            )
        }

    except Exception as e:
        return {
            "error": str(e)
        }