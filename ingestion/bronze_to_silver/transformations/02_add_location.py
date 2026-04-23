import dlt
from pyspark.sql.functions import (
    col, to_date, date_format, trim, initcap, create_map,
    split, size, when, concat, lit, abs, to_timestamp, regexp_extract
)
from itertools import chain

catalogo = "workspace"
base_schema = "turismo"

mapa_estados = {
    "aqua_rio_rj": "RJ",
    "beto_carreiro_sc": "SC",
    "elevador_lacerda": "BA",
    "hopi_hari": "SP",
    "jardim_botanico": "RJ",
    "mercado_central": "MG",
    "mercado_ver_o_peso": "PA",
    "museu_arte_sp": "SP",
    "museu_imperial": "RJ",
    "parque_chapada": "GO",
    "parque_jalapao": "TO",
    "parque_nacional": "RJ",
    "pelourinho_ba": "BA",
    "praca_3_poderes": "DF",
}

mapping_expr = create_map([lit(x) for x in chain(*mapa_estados.items())])
bronze_tables = spark.sql(f"SHOW TABLES IN {catalogo}.{base_schema} LIKE 'bronze*'")
nomes_tabelas = [row["tableName"] for row in bronze_tables.collect()]

selects = [
    f"SELECT *, '{nome.replace('bronze_', '', 1)}' AS name "
    f"FROM {catalogo}.{base_schema}.{nome}"
    for nome in nomes_tabelas
]
union_query = "\nUNION ALL\n".join(selects)

# --- Tabela silver final com agrupamento e a coluna de qual estado é a avaliação ---
@dlt.table(
    name="silver_turismos",
    comment="União das bronze com coluna de estado",
    table_properties={"quality": "silver"}
)
def silver_turismos():
      df = spark.sql(union_query).withColumn("estado", mapping_expr[col("name")])
      return df
    