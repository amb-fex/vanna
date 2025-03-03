from .pgvector.pgvector import PG_VectorStore
from .modelo_amb.modelo_amb import ModeloAMB
import sentencepiece
#from .hf import Hf

class AmbVanna(PG_VectorStore, ModeloAMB):
    def __init__(self, config=None):
        PG_VectorStore.__init__(self, config=config)
        ModeloAMB.__init__(self, config=config)
