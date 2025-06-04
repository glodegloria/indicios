import pandas as pd
import re
from datetime import datetime

# Rutas
input_path = r"Mollusca.tsv"
output_path = r"Occurrence_filtrado_F1_mollusca_sinprofundidad.txt"

# Col
columnas_ordenadas = [
    "decimalLatitude", 
    "decimalLongitude", 
    "depth", 
    "genusid", 
    "fecha_hora"  # aqui has de agregar las columnas que deseas recojer
]

# Formatos conocidos
formatos_fecha = {
    '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S', '%d/%m/%Y',
    '%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%Y-%m-%dT%H:%M%z',
    '%Y-%m', '%Y', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y',
}

# Variables (ponerchunks para que no pete)
chunk_size = 200000
total_rows = 0
filtradas_total = 0
sin_fecha_total = 0
eliminadas_por_profundidad = 0
chunks_filtrados = []

# Funcion de carga data de muestreo
def parse_eventDate(eventDate):
    if pd.isna(eventDate) or not isinstance(eventDate, str):
        return None
    eventDate = eventDate.strip()
    for fmt in formatos_fecha:
        try:
            dt = datetime.strptime(eventDate, fmt)
            return dt.strftime('%d-%m-%Y %H:%M:%S') if dt.time() != datetime.min.time() else dt.strftime('%d-%m-%Y')
        except:
            continue
    # Buscar patrón flexible (YYYY-MM-DD HH:MM:SS)
    match = re.search(r'(\d{4}-\d{2}-\d{2})([ T](\d{2}:\d{2}:\d{2}))?', eventDate)
    if match:
        try:
            full_str = match.group(0)
            dt = datetime.strptime(full_str.strip(), '%Y-%m-%d %H:%M:%S') if match.group(2) else datetime.strptime(match.group(1), '%Y-%m-%d')
            return dt.strftime('%d-%m-%Y %H:%M:%S') if match.group(2) else dt.strftime('%d-%m-%Y')
        except:
            return None
    return None
#for chunk in pd.read_csv(input_path, sep='\t', encoding='utf-8', chunksize=chunk_size, engine='python'):
for chunk in pd.read_csv(input_path, sep='\t', encoding='utf-8',  chunksize=chunk_size, engine='python'):
    total_rows += len(chunk)
    chunk = chunk.copy()

    print("Hace cosas")


    
    # Profundidad válida (filtrahe segun la profundidad, a partir de los 4000 se supone que no es placa contienental)
    #if 'depth' in chunk.columns:
    #    chunk["depth"] = pd.to_numeric(chunk["depth"], errors="coerce")
    #    antes_depth = len(chunk)
    #    chunk = chunk[(chunk["depth"] >= 0) & (chunk["depth"] <= 4000)]
    #    eliminadas_por_profundidad += (antes_depth - len(chunk))

    # Convertir columnas a numérico
    for col in ["year", "month", "day"]:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    if "year" in chunk.columns:
        chunk = chunk[chunk["year"].between(1950, 2025)]

    # Crear columna "fecha_hora"
    fechas = []
    for idx, row in chunk.iterrows():
        fecha = None
        if all(col in row and pd.notna(row[col]) for col in ["year", "month", "day"]):
            try:
                dt = datetime(int(row["year"]), int(row["month"]), int(row["day"]))
                fecha = dt.strftime('%d-%m-%Y')
            except:
                pass
        elif "eventDate" in row and pd.notna(row["eventDate"]):
            fecha = parse_eventDate(str(row["eventDate"]))
        fechas.append(fecha)

    chunk["fecha_hora"] = fechas

    # Eliminar filas sin fecha_hora
    antes = len(chunk)
    chunk = chunk[chunk["fecha_hora"].notna()]
    sin_fecha_total += (antes - len(chunk))

    # Asegurar columnas ordenadas
    for col in columnas_ordenadas:
        if col not in chunk.columns:
            chunk[col] = pd.NA
    chunk = chunk[columnas_ordenadas]

    filtradas_total += len(chunk)
    chunks_filtrados.append(chunk)

# Concatenar
df_final = pd.concat(chunks_filtrados, ignore_index=True)

# Guardar
df_final.to_csv(output_path, sep='\t', index=False)

# Reporte
print(f"Archivo guardado en: {output_path}")
print(f"Filas procesadas en total: {total_rows}")
print(f"Filas después del filtrado: {filtradas_total}")
print(f"Filas eliminadas por no contener fecha válida: {sin_fecha_total}")
print(f"Filas eliminadas por valores inválidos de profundidad (fuera de 0-4000 m): {eliminadas_por_profundidad}")
