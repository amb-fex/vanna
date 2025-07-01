

from vanna.modelo_amb.modelo_amb import ModeloAMB
from .vector_store import AMB_VectorStore
import torch
from datetime import datetime

class AmbVannaCodigo(ModeloAMB, AMB_VectorStore):
    def __init__(self, config=None):
        super().__init__(config=config)
        AMB_VectorStore.__init__(self, config=config)
        print("Modelo general usa el modelo base tal cual.")
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
     
    Envía un prompt al modelo de lenguaje y devuelve la resposta generada.
        
        """
      
        if prompt[0]["role"] == "system":
            system_msg = prompt[0]["content"]
            # Pegamos esta instrucción al principio de la primera 'user'
            for i in range(1, len(prompt)):
                if prompt[i]["role"] == "user":
                    prompt[i]["content"] = system_msg + "\n\n" + prompt[i]["content"]
                    break
            prompt = prompt[1:]  # Eliminamos el 'system' ya que ya lo usamos

        #  Mostrem el prompt abans de la generació
        print("=== Prompt tokens decodificats ===")
        for p in prompt:
            print(f"{p['role']}: {p['content']}")

        # Apliquem el format de plantilla segons el model
        full_prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            date_string=datetime.today().strftime("%Y-%m-%d")
        )


        # Codifiquem el prompt
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.model.device)

        print("\n==== PROMPT REAL AL MODELO CODIGO ====\n", full_prompt)
        print("\n==== LONGITUD (tokens) ====\n", len(input_ids[0]))

        # Generem la resposta
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=3000,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.2,
            top_p=1.0,
        )

        # Extraiem només la part generada
        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)

        self.log(response)

        return response
    
    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}

    
