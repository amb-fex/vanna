from ..pgvector import PG_VectorStore

# 🔹 Database Configuration for Supabase
config = {
    "connection_string": "postgresql://postgres.spdwbcfeoefxnlfdhlgi:chatbot2025@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?options=-csearch_path=vector_store"
}

# 🔹 Create a subclass to implement missing abstract methods
class CustomVectorStore(PG_VectorStore):
    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        return "This is a placeholder response from the model."

# 🔹 Instantiate CustomVectorStore instead of PG_VectorStore
vector_store = CustomVectorStore(config)
