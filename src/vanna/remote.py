import dataclasses
import json
from io import StringIO
from typing import Callable, List, Tuple, Union
from datetime import datetime

import pandas as pd
import requests

from .base import VannaBase
from .types import (
  AccuracyStats,
  ApiKey,
  DataFrameJSON,
  DataResult,
  Explanation,
  FullQuestionDocument,
  NewOrganization,
  NewOrganizationMember,
  Organization,
  OrganizationList,
  PlotlyResult,
  Question,
  QuestionCategory,
  QuestionId,
  QuestionList,
  QuestionSQLPair,
  QuestionStringList,
  SQLAnswer,
  Status,
  StatusWithId,
  StringData,
  TrainingData,
  UserEmail,
  UserOTP,
  Visibility,
)
from .vannadb import VannaDB_VectorStore

import plotly
import plotly.graph_objs as go
import traceback
from typing import Union, Tuple


class VannaDefault(VannaDB_VectorStore):
    def __init__(self, model: str, api_key: str, config=None):
        VannaBase.__init__(self, config=config)
        VannaDB_VectorStore.__init__(self, vanna_model=model, vanna_api_key=api_key, config=config)

        self._model = model
        self._api_key = api_key

        self._endpoint = (
            "https://ask.vanna.ai/rpc"
            if config is None or "endpoint" not in config
            else config["endpoint"]
        )
    
    def get_sql_prompt(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        Genera un prompt para que el modelo LLM cree consultas SQL, enfocándose específicamente en el contexto del geoportal de cartografía del AMB

        Example:
        ```python
        vn.get_sql_prompt(
            question="Quin és el número de descàrregues totals durant l’any 2024?",
            question_sql_list=[{"question": "Quin és el nombre de descàrregues totals?", "sql": "SELECT COUNT(*) FROM public.descargas;"}],
            ddl_list=["CREATE TABLE public.descargas (id INT, idfull VARCHAR, fechadescarga TIMESTAMP, ...);"],
            doc_list=["La taula descargas conté informació sobre les descàrregues d'arxius cartogràfics."]
        )
        ```

        Args:
            initial_prompt (str): The base instruction for the system message. If None, a domain-specific one will be generated.
            question (str): The user question to generate SQL for.
            question_sql_list (list): A list of previous question-SQL examples to guide the LLM.
            ddl_list (list): List of DDL statements (table schemas) relevant to the question.
            doc_list (list): List of documentation snippets related to the tables or domain.

        Returns:
            list: A message_log (prompt) ready to be sent to the LLM to generate a SQL query.
        """

        # prompt inicial
        if initial_prompt is None:
            initial_prompt = (
                "Ets un expert en PostgreSQL en bases de dades de cartografia municipal i estadística d'ús d'un geoportal."
                "Si us plau, ajuda a generar una consulta SQL per respondre la pregunta. La teva resposta ha d’estar BASADA ÚNICAMENT en el context proporcionat i ha de seguir les directrius de resposta i les instruccions de format. "
                "No pots utilitzar coneixement extern. "
                "Genera una consulta PostgreSQL correcta basada exclusivament en aquest context.\n"
                "No facis servir les taules 'langchain_pg_embedding' ni 'langchain_pg_collection' ni 'spatial_ref_sys' ni 'stored_charts', ja que no contenen informació rellevant per a l'analítica del geoportal.\n"
            )
             

   
          # 1. Agregar DDL (estructura de tablas)
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        # 2. Agregar documentación
        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )
   
             # Si se pasa un error anterior, se añade al contexto del prompt
        if "error_feedback" in kwargs:
            initial_prompt += f"\n===Error anterior detectat:\n{kwargs['error_feedback']}\n"
  

        # 3. Construir el message_log
        message_log = [self.system_message(initial_prompt)]

        # Añadir ejemplos anteriores si existen
        for example in question_sql_list:
            if not question_sql_list:
                print("no hay ejemplos previos")
            if example is not None and "question" in example and "sql" in example:
                message_log.append(self.user_message(example["question"]))
                message_log.append(self.assistant_message(example["sql"]))

        # Finalmente añadir la nueva pregunta del usuario
        message_log.append(self.user_message(question))

        return message_log



    
    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}

    def extract_sql_query(self, text: str) -> str:
        """
        Extrae la primera consulta SQL válida después de 'SELECT',
        eliminando caracteres no deseados.
        """
        sql = super().extract_sql(text)
        return sql.replace("\\_", "_").replace("\\", "")
    
    def corregir_sql_any(self, sql: str) -> str:
        """
        Corrige el uso incorrecto de 'any' como alias de año.
        """
        # Casos comunes de alias
        sql = sql.replace(" AS any", " AS y")
        sql = sql.replace(" as any", " AS y")
        sql = sql.replace(" BY any", " BY y")
        sql = sql.replace(" by any", " BY y")
    
        # Casos en SELECT, GROUP BY, etc.
        sql = sql.replace("SELECT any", "SELECT y")
        sql = sql.replace(", any", ", y")
        sql = sql.replace(" any,", " y,")
        sql = sql.replace(" any ", " y ")
        sql = sql.replace("any;", " y; ")



    def submit_prompt(self, prompt, **kwargs) -> str:
      # JSON-ify the prompt
      json_prompt = json.dumps(prompt, ensure_ascii=False)
  
      params = [StringData(data=json_prompt)]
  
      d = self._rpc_call(method="submit_prompt", params=params)
  
      if "result" not in d:
          return None
  
      # Load the result into a dataclass
      results = StringData(**d["result"])
  
      return results.data
    
    def generate_sql(self, question: str, **kwargs) -> str:
        # Usa la función base de Vanna
        sql = super().generate_sql(question, **kwargs)

        # Limpiezas comunes
        sql = sql.replace("\\_", "_")
        sql = sql.replace("\\", "")

        # Corrección de errores comunes como 'any' mal usado
        sql = self.corregir_sql_any(sql)

      

        self.log(sql)

        return sql
    
