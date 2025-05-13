import dataclasses
import json
from io import StringIO
from typing import Callable, List, Tuple, Union

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
import pandas as pd
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
    
    def ask(
      self,
      question: Union[str, None] = None,
      print_results: bool = True,
      auto_train: bool = True,
      visualize: bool = True,
      allow_llm_to_see_data: bool = False,
  ) -> Union[
      Tuple[
          Union[str, None],
          Union[pd.DataFrame, None],
          Union[plotly.graph_objs.Figure, None],
      ],
      None,
  ]:
      if question is None:
          question = input("Enter a question: ")
  
      comentario_sql = None
  
      try:
          sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
  
          # Iteración sobre comentarios de SQL
          refinado = True
          while refinado:
              refinado = False
              respuesta = input("¿Quieres comentar o corregir la SQL generada? (s/n): ").strip().lower()
              if respuesta == "s":
                  comentario_sql = input("Escribe tu comentario o corrección:\n")
                  if comentario_sql:
                      question = self.generate_rewritten_question(question, comentario_sql)
                      sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
                      if print_results:
                          print("SQL actualizada:\n", sql)
                      refinado = True
  
      except Exception as e:
          print(e)
          return None, None, None
  
      if print_results:
          try:
              Code = __import__("IPython.display", fromList=["Code"]).Code
              display(Code(sql))
          except Exception:
              print(sql)
  
      if self.run_sql_is_set is False:
          print("If you want to run the SQL query, connect to a database first.")
          return (None if print_results else (sql, None, None))
  
      try:
          df = self.run_sql(sql)
  
          if print_results:
              try:
                  display = __import__("IPython.display", fromList=["display"]).display
                  display(df)
              except Exception:
                  print(df)
  
          if len(df) > 0 and auto_train:
              self.add_question_sql(question=question, sql=sql)
  
          if visualize:
              comentario_plot = None
              try:
                  plotly_code = self.generate_plotly_code(
                      question=question,
                      sql=sql,
                      df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                  )
                  fig = self.get_plotly_figure(plotly_code=plotly_code, df=df)
  
                  # Iteración sobre mejoras del gráfico
                  refinado_plot = True
                  while refinado_plot:
                      refinado_plot = False
                      respuesta_plot = input("¿Quieres hacer cambios en el gráfico? (s/n): ").strip().lower()
                      if respuesta_plot == "s":
                          comentario_plot = input("Describe el cambio que quieres (e.g. colores, leyenda...):\n")
                          if comentario_plot:
                              plotly_code = self.submit_prompt([
                                  self.system_message(f"El DataFrame es:\n{df.head(10).to_markdown()}"),
                                  self.user_message(f"Código actual de la gráfica:\n```python\n{plotly_code}\n```\nQuiero hacer este cambio: {comentario_plot}. Devuélveme solo el nuevo código Plotly.")
                              ])
                              plotly_code = self._sanitize_plotly_code(self._extract_python_code(plotly_code))
                              fig = self.get_plotly_figure(plotly_code=plotly_code, df=df)
                              refinado_plot = True
  
                  if print_results:
                      try:
                          display = __import__("IPython.display", fromlist=["display"]).display
                          Image = __import__("IPython.display", fromlist=["Image"]).Image
                          img_bytes = fig.to_image(format="png", scale=2)
                          display(Image(img_bytes))
                      except Exception:
                          fig.show()
  
                  return sql, df, fig
  
              except Exception as e:
                  traceback.print_exc()
                  print("Couldn't run plotly code: ", e)
                  return (None if print_results else (sql, df, None))
          else:
              return sql, df, None
  
      except Exception as e:
          print("Couldn't run sql: ", e)
          return (None if print_results else (sql, None, None))

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

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
