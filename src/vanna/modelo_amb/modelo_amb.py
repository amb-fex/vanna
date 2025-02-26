import re 
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..base import VannaBase
from .vector_store import vector_store

# Esta clase se trata de un archivo constumisa del hf (modelo hugging face) en vanna. 

class ModeloAMB(VannaBase):
    def __init__(self, config=None):
        if config is None:
            config = {}

        super().__init__(config)  # Inicializa VannaBase

        model_name_or_path = config.get("model_name_or_path")
        quantization_config = config.get("quantization_config", None)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
        )

        # Conectar con PG_VectorStore
        #self.vector_store = vector_store

    #  Implementación de los métodos abstractos requeridos por VannaBase

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
        """
        Genera SQL utilizando el LLM y almacena la consulta en PG_VectorStore.
        Antes de generar, revisa si hay una consulta similar en la base vectorial.
        """
        try:
            # Buscar en PG_VectorStore si existe una consulta similar
            similar_sqls = self.vector_store.get_similar_question_sql(question)
            if similar_sqls:
                return similar_sqls[0]["sql"]  # Devuelve la primera coincidencia

            # Si no hay consultas previas, generamos una nueva con el LLM
            prompt = f"Genera una consulta SQL para la siguiente pregunta:\n{question}"
            sql = self.submit_prompt(prompt)
            sql = self.extract_sql_query(sql)

            # Almacenar la nueva consulta SQL en PG_VectorStore
            self.vector_store.add_question_sql(question, sql)

            return sql

        except Exception as e:
            print(f" Error en 'generate_sql': {e}")
            return "Error al generar la consulta SQL."

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
