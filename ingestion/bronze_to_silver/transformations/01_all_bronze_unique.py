import dlt
from pyspark.sql.functions import (
    col, to_date, date_format, trim, initcap,
    split, size, when, concat, lit, abs, to_timestamp, regexp_extract
)

catalogo = "workspace"
base_schema = "turismo"

# 1. Lista as tabelas bronze (executa no momento da definição do pipeline)
bronze_tables = spark.sql(f"SHOW TABLES IN {catalogo}.{base_schema} LIKE 'bronze*'")
nomes_tabelas = [row["tableName"] for row in bronze_tables.collect()]

# 2. Monta um SELECT para cada tabela, já adicionando a coluna 'name'
#    com o nome do local (sem o prefixo 'bronze_')
selects = [
    f"SELECT *, '{nome.replace('bronze_', '', 1)}' AS name "
    f"FROM {catalogo}.{base_schema}.{nome}"
    for nome in nomes_tabelas
]
union_query = "\nUNION ALL\n".join(selects)


@dlt.table(
    name="silver_avaliacoes",
    comment="União de todas as tabelas bronze de avaliações turísticas, com coluna 'name' identificando o local",
    table_properties={
        "quality": "silver"
    }
)
def silver_avaliacoes():
    return spark.sql(union_query)