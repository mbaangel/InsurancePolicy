import pandas as pd

def text_to_dataframe(docs_default):
    # Extraer el contenido de los documentos en una lista de textos usando la clave content
    texts = [doc["content"] for doc in docs_default] 

    # Crear DataFrame con los textos
    df = pd.DataFrame({"text": texts})

    # Agregar estadísticas de texto
    df["num_words"] = df["text"].apply(lambda x: len(x.split()))
    df["num_chars"] = df["text"].apply(lambda x: len(x))

    # Mostrar estadísticas generales
    print(df[["num_words", "num_chars"]].describe())  # Muestra distribución
    print(f"Total words: {df['num_words'].sum()}")  # Suma total de palabras
    print(f"Total chars: {df['num_chars'].sum()}")  # Suma total de caracteres

    return df