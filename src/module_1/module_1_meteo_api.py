import time
import requests
import logging
import json
import pandas as pd
from urllib.parse import urlencode
import matplotlib.pyplot as plt

# Configuramos un logger para ver los avisos por consola 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constantes del enunciado
API_URL = "https://archive-api.open-meteo.com/v1/archive?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

def _request_with_cooloff(url: str, headers: dict, num_attempts: int, payload: dict | None = None) -> requests.Response:    
    # Funcion interna para hacer la peticion. Si la API da error por saturacion, 
    # hace reintentos esperando cada vez mas tiempo (cooloff)
    cooloff = 1
    for call_count in range(num_attempts):
        try:
            if payload is None:
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Error de conexion: {e}")
            if call_count != (num_attempts - 1):
                time.sleep(cooloff)
                cooloff *= 2 # Doblamos el tiempo de espera para el siguiente intento
                continue
            raise

        except requests.exceptions.HTTPError as e:
            # Si es un error 404 (no encontrado), no tiene sentido reintentar
            if response.status_code == 404:
                raise
            
            logger.info(f"API devolvio codigo {response.status_code}, esperamos {cooloff}s")
            if call_count != (num_attempts - 1):
                time.sleep(cooloff)
                cooloff *= 2
                continue
            raise

def request_wrapper(url: str, headers: dict = {}, payload: dict | None = None) -> dict:    
    # Helper que llama a la funcion de arriba y ya nos devuelve el JSON listo para usar
    res = _request_with_cooloff(url, headers, num_attempts=5, payload=payload)
    return json.loads(res.content.decode("utf-8"))

def get_data_meteo_api(longitude: float, latitude: float, start_date: str, end_date: str) -> dict:
    # Prepara los parametros y construye la url final para descargar los datos de una ciudad
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(VARIABLES),
    }
    full_url = API_URL + urlencode(params, safe=",")
    return request_wrapper(full_url)

def compute_monthly_statistics(data: pd.DataFrame, meteo_variables: list[str]) -> pd.DataFrame:
    # Coge los datos diarios, los agrupa por mes y calcula estadisticas clave
    data["time"] = pd.to_datetime(data["time"])
    
    # Agrupamos combinando la ciudad y el mes/año
    grouped = data.groupby([data["city"], data["time"].dt.to_period("M")])
    
    results = []
    for (city, month), group in grouped:
        stats = {"city": city, "month": month.to_timestamp()}
        
        # Para cada variable (temp, lluvia, viento) sacamos sus metricas
        for var in meteo_variables:
            stats[f"{var}_max"] = group[var].max()
            stats[f"{var}_mean"] = group[var].mean()
            stats[f"{var}_min"] = group[var].min()
            stats[f"{var}_std"] = group[var].std()
            
        results.append(stats)

    return pd.DataFrame(results)

def plot_timeseries(data: pd.DataFrame) -> None:    
    # Pinta los graficos creando una cuadricula: filas = variables, columnas = ciudades
    rows = len(VARIABLES)
    cols = len(data["city"].unique())
    
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 6 * rows))

    for i, var in enumerate(VARIABLES):
        for k, city in enumerate(data["city"].unique()):
            city_data = data[data["city"] == city]
            ax = axs[i, k]

            # 1. Pintamos la linea de la media
            ax.plot(city_data["month"], city_data[f"{var}_mean"], 
                   label=f"{city} (media)", color=f"C{k}")

            # 2. Area sombreada entre el minimo y el maximo de ese mes
            ax.fill_between(city_data["month"], 
                          city_data[f"{var}_min"], 
                          city_data[f"{var}_max"], 
                          alpha=0.2, color=f"C{k}")

            # 3. Barras de error indicando la desviacion estandar
            ax.errorbar(city_data["month"], city_data[f"{var}_mean"], 
                       yerr=city_data[f"{var}_std"], fmt="none", 
                       ecolor=f"C{k}", alpha=0.5)

            ax.set_title(var)
            ax.legend()

    plt.tight_layout()
    # Guardamos la imagen directamente en la carpeta
    plt.savefig("src/module_1/climate_evolution.png", bbox_inches="tight")

def main() -> None:    
    all_data = []
    start = "2010-01-01"
    end = "2020-12-31"

    # 1. Bucle para descargar los datos de todas las ciudades
    for city, coords in COORDINATES.items():
        print(f"Descargando datos de {city}...")
        raw = get_data_meteo_api(coords["longitude"], coords["latitude"], start, end)
        
        df = pd.DataFrame(raw["daily"])
        df["city"] = city # Super importante añadir esto antes de juntarlos
        all_data.append(df)

    # 2. Juntamos todos los DataFrames y calculamos las estadisticas mensuales
    full_df = pd.concat(all_data)
    monthly_df = compute_monthly_statistics(full_df, VARIABLES)

    # 3. Exportamos los datos a CSV y generamos el grafico final
    monthly_df.to_csv("src/module_1/meteo_stats.csv")
    plot_timeseries(monthly_df)
    print("Proceso terminado. Grafico guardado en src/module_1/")

if __name__ == "__main__":
    main()