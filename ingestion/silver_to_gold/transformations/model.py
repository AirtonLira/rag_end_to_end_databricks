import mlflow
import mlflow.pyfunc
import pandas as pd
import requests
import json
import time
import re


class ClassificadorComentarios(mlflow.pyfunc.PythonModel):
    """
    Classifica comentários turísticos em: elogio, critica, duvida, outros.
    Usa OpenRouter com Llama 3.1 8B gratuito, processando em lotes.
    """

    CATEGORIAS_VALIDAS = {"elogio", "critica", "duvida", "outros"}
    TAMANHO_LOTE = 10

    def load_context(self, context):
        with open(context.artifacts["config"], "r") as f:
            config = json.load(f)
        self.api_key = config["api_key"]
        self.modelo = "openrouter/free"
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def _montar_prompt(self, descricoes):
        comentarios = "\n".join(f"{i+1}. {d}" for i, d in enumerate(descricoes))
        return f"""Você é um classificador de comentários turísticos. Classifique cada comentário em UMA das categorias:

- elogio: expressa satisfação, aprovação ou positividade
- critica: expressa insatisfação, reclamação ou negatividade
- duvida: é uma pergunta ou pedido de informação
- outros: não se encaixa nas três anteriores

Responda APENAS no formato "número. categoria", uma por linha, sem explicações.

Comentários:
{comentarios}

Respostas:"""

    def _chamar_api(self, prompt, tentativa=0):
        try:
            resp = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.modelo,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0,
                },
                timeout=60,
            )
            if resp.status_code == 429 and tentativa < 3:
                espera = 2 ** (tentativa + 3)
                print(f"Rate limit. Aguardando {espera}s...")
                time.sleep(espera)
                return self._chamar_api(prompt, tentativa + 1)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Erro na API: {e}")
            return ""

    def _parsear_resposta(self, texto, qtd_esperada):
        resultado = ["outros"] * qtd_esperada
        padrao = re.compile(r"(\d+)[.\):\-\s]+(\w+)")
        for match in padrao.finditer(texto.lower()):
            idx = int(match.group(1)) - 1
            categoria = (
                match.group(2)
                .replace("í", "i").replace("ú", "u")
                .replace("ó", "o").replace("á", "a").replace("ê", "e")
            )
            if 0 <= idx < qtd_esperada and categoria in self.CATEGORIAS_VALIDAS:
                resultado[idx] = categoria
        return resultado

    def _classificar_lote(self, descricoes):
        limpas = [str(d) if d and isinstance(d, str) and d.strip() else "" for d in descricoes]
        if not any(limpas):
            return ["outros"] * len(descricoes)
        prompt = self._montar_prompt(limpas)
        resposta = self._chamar_api(prompt)
        categorias = self._parsear_resposta(resposta, len(limpas))
        time.sleep(1)
        return categorias

    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
        descricoes = model_input["descricao"].tolist()
        resultados = []
        for i in range(0, len(descricoes), self.TAMANHO_LOTE):
            resultados.extend(self._classificar_lote(descricoes[i:i + self.TAMANHO_LOTE]))
        return pd.Series(resultados)
    
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
import json
import os

# Aponta o MLflow pro Unity Catalog (em vez do workspace registry antigo)
mlflow.set_registry_uri("databricks-uc")

# Salva a config num arquivo temporário (API key + modelo escolhido)
config_path = "/tmp/openrouter_config.json"
with open(config_path, "w") as f:
    json.dump({
        "api_key": dbutils.secrets.get(scope="turismo", key="openrouter_key"),
        "modelo": "openrouter/free",   # troque conforme sua escolha
    }, f)

# Assinatura do modelo (ajuda o MLflow a validar entradas)
exemplo_input = pd.DataFrame({"descricao": ["O lugar é maravilhoso, adorei!"]})
exemplo_output = pd.Series(["elogio"])
signature = infer_signature(exemplo_input, exemplo_output)

catalogo = "workspace"
schema = "turismo"
nome_modelo = f"{catalogo}.{schema}.classificador_comentarios"

with mlflow.start_run(run_name="classificador_comentarios_v1"):
    mlflow.pyfunc.log_model(
        artifact_path="modelo",
        python_model=ClassificadorComentarios(),
        artifacts={"config": config_path},
        pip_requirements=["requests", "pandas", "mlflow"],
        signature=signature,
        input_example=exemplo_input,
        registered_model_name=nome_modelo,
    )
