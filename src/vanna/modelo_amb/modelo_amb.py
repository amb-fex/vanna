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

  

