import pandas as pd
# Importamos la funcion y las variables de tu script principal
from src.module_1.module_1_meteo_api import compute_monthly_statistics, VARIABLES

def test_compute_monthly_statistics():
    # 1. PREPARACION: Creamos unos datos falsos de prueba (2 dias de Enero, 1 de Febrero)
    fake_data = {
        "time": ["2020-01-01", "2020-01-15", "2020-02-10"],
        "city": ["Madrid", "Madrid", "Madrid"],
        "temperature_2m_mean": [10.0, 12.0, 15.0],
        "precipitation_sum": [0.0, 5.0, 0.0],
        "wind_speed_10m_max": [10.0, 15.0, 20.0]
    }
    df = pd.DataFrame(fake_data)

    # 2. EJECUCION: Le pasamos los datos falsos a tu funcion
    result = compute_monthly_statistics(df, VARIABLES)

    # 3. COMPROBACION (Asserts): Comprobamos que hace lo que creemos que hace
    
    # Como habia dias de enero y febrero, el resultado debe tener 2 filas (2 meses)
    assert len(result) == 2 
    
    # La media de temperatura de Enero deberia ser 11.0 (porque (10 + 12) / 2 = 11)
    assert result.iloc[0]["temperature_2m_mean_mean"] == 11.0
    
    # Comprobamos que el maximo de lluvia de Enero fue 5.0
    assert result.iloc[0]["precipitation_sum_max"] == 5.0