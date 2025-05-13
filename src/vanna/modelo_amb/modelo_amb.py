import sentencepiece
import re 
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..base import VannaBase
from .vector_store import vector_store


class ModeloAMB(VannaBase):
    def __init__(self, config=None):
        if config is None:
            config = {}

        super().__init__(config)  # Inicializa VannaBase

        model_name_or_path = config.get("model_name_or_path")
        token = config.get("token")
        quantization_config = config.get("quantization_config", {})

        # Carga del tokenizer con o sin token
        if token:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=token)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Carga del modelo con manejo de cuantización si es necesario
        model_params = {
            "device_map": "auto",
            "use_auth_token": token if token else None
        }

        # Agregar configuraciones de cuantización solo si están definidas
        model_params.update(quantization_config) if quantization_config else None

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_params)
    
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

    def generate_sql(self, question: str, **kwargs) -> str:
        # Use the super generate_sql
        sql = super().generate_sql(question, **kwargs)

        # Replace "\_" with "_"
        sql = sql.replace("\\_", "_")

        sql = sql.replace("\\", "")

        return self.extract_sql_query(sql)

    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        Envía un prompt al modelo de lenguaje y devuelve la respuesta generada.
        """
        input_ids = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=1,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        self.log(response)

        return response

        # Conectar con PG_VectorStore
        #self.vector_store = vector_store

    #  Implementación de los métodos abstractos requeridos por VannaBase
'''

    def generate_embedding(self, data: str, **kwargs) -> list:
        """
        Genera embeddings para almacenar consultas en la base de datos vectorial.
        """
        return []  # Aquí puedes integrar un modelo de embeddings real si es necesario.

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        Obtiene SQLs similares desde PG_VectorStore.
        """
        return self.vector_store.get_similar_question_sql(question)

    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        Obtiene DDLs relacionados desde PG_VectorStore.
        """
        return self.vector_store.get_related_ddl(question)

    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        Obtiene documentación relacionada desde PG_VectorStore.
        """
        return self.vector_store.get_related_documentation(question)

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        Agrega una consulta SQL al almacenamiento vectorial.
        """
        return self.vector_store.add_question_sql(question, sql)

    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        Agrega una declaración DDL al almacenamiento vectorial.
        """
        return self.vector_store.add_ddl(ddl)

    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        Agrega documentación a la base de datos vectorial.
        """
        return self.vector_store.add_documentation(documentation)

    def get_training_data(self, **kwargs) -> list:
        """
        Obtiene los datos de entrenamiento almacenados en PG_VectorStore.
        """
        return self.vector_store.get_training_data()

    def remove_training_data(self, id: str, **kwargs) -> bool:
        """
        Elimina un conjunto de datos de entrenamiento del almacenamiento vectorial.
        """
        return self.vector_store.remove_training_data(id)
      
    '''
