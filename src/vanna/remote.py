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
             




        # 1. Agregar instrucciones claras la satatic_docu respuesta
        initial_prompt += (
            "===Directrius de resposta\n"
            "1. Si el context proporcionat és suficient, genera una consulta SQL sense cap explicació.\n"
            "2. Si el context és gairebé suficient però falta una cadena específica, genera una consulta SQL intermèdia comentada com 'intermediate_sql'.\n"
            "3. Assegura't que les funcions SQL com ROUND(...) tanquin correctament els parèntesis i que l’ús de AS sigui sintàcticament correcte.\n"
            "4. Si el context no és suficient, indica-ho explícitament.\n"
            "5. Fes servir les taules més rellevants.\n"
            "6. Si la pregunta ja té resposta, repeteix la resposta exacta.\n"
            f"7. Assegura que la sortida SQL sigui compatible amb {self.dialect}, executable i sense errors de sintaxi.\n"
            "8. Només pots respondre generant una consulta SQL o indicant explícitament que no pots generar-la. No pots escriure missatges de conversa, salutacions o comentaris personals.\n"
            "9. En subconsultes (CTE o SELECT dins d’un WITH), utilitza només les columnes disponibles en la subconsulta immediata. No reutilitzis àlies de taula (com `u` o `d`) si no han estat definits explícitament en aquest nivell.\n"

"""

    TAULES I ATRIBUTS RELLEVANTS:

Taula: descargas
• `fechadescarga`: Data i hora de la descàrrega.
• `nomproducte`: Nom exacte del producte descarregat.
• `idcategoria`: Categoria cartogràfica del producte.
• `format`: Format del fitxer descarregat.
• `usuario`: Identificador de l’usuari que ha descarregat.
• `idepoca`: Temporalitat de la informació.
• `nombrefichero`: el nombre del fichero que descarga el usuario, para algunas categorias (CartografiaMarina, GuiaMetropolitana, Lidar, OrtofotoPlatges) y sus respectivos productos (Cartografia topobatimètrica, Model d’elevacions topobatimètric, Batimetria multifeix, Lidar platges 2017, Orto Platges 2012, etc) el nombre puede contener inforcion importante como el municipio
• `idfull`: Full geogràfic relacionat amb la geometria.
• `geom`: Geometria espacial (tipus MULTIPOLYGON, en EPSG:4326) corresponent al full descarregat.
• `centroid_geom`  Centreide del full descarregat (tipus POINT, en EPSG:4326)

Taula: usuarios
• `usuario`: Clau primària de l’usuari.
• `genero`: Gènere. Valors: "Mujer", "Hombre", "Otros", "No respondido".
• `nomperfil`: Perfil d’usuari. Valors: "Acadèmic", "Altres", "Privat", "Públic".
• `nomambito`: Àmbit professional. Exemples: "Arquitectura", "Medi ambient", etc.
• `procedencia`: Origen institucional. Valors: "CARTOGRAFIA", "PLANEJAMENT".
• `idioma`: Idioma de la interfície. Valors: "Català", "Castellà", "Anglès".
• `ididioma`: Codi de llengua. Valors: "ca", "es", "en".

Taula: click
• `fecha`: Data del clic.
• `idcategoria`, `nomproducte`: Igual que a descargas.
• `idioma`: Idioma de la interfície. Valors: "Català", "Castellà", "Anglès".
• `lat`, `lon`: Coordenades geogràfiques.
• `geom`: Geometria del punt del clic (tipus POINT, EPSG:4326), creada a partir de lat i lon, utilitzada per fer consultes espacials com la relació amb districtes o municipis.

Taula: fulls
• `idfull`: Identificador del full.
• `idcategoria`, `nomproducte`: Igual que a descargas.
• `geom`: Geometria espacial (polígon).
• `usuario`, igual que a taula usuarios

Taula: districtes_barcelona
• `id`: Identificador.
• `geom`: Geometria del districte (polígon).
• `nom`: Nom del districte.
• `codi_ua`: Codi administratiu.
• `area`: Superfície.

Taula: divisions_administratives
• `id`: Identificador.
• `geom`: Geometria del municipi.
• `codimuni`: Codi de municipi.
• `nommuni`: Nom del municipi.
• `aream5000`: Superfície.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERPRETACIÓ FLEXIBLE DE VALORS INTRODUÏTS PER L'USUARI:

Quan l’usuari introdueix un nom de producte, categoria, format, idioma o classificació que **no coincideix exactament** amb els valors vàlids, el sistema ha de:

1. Cercar el valor més semblant dins del conjunt de valors vàlids per aquell camp.
2. Substituir automàticament el valor incorrecte pel valor més proper si la coincidència és clara.
3. Generar la consulta SQL utilitzant **només valors vàlids**, fins i tot si la forma escrita original de l’usuari era diferent.

Aquest comportament s’aplica als següents camps:
- `nomproducte` (productes)
- `idcategoria` (categories)
- `format` (formats de fitxer)
- `idioma` (idioma de la interfície)
- `nomperfil`, `nomambito`, `procedencia` (classificació d’usuaris)

Exemples:
- Si l’usuari escriu "ortofoto platges", el sistema ha de reconèixer que probablement es refereix a `"OrtofotoPlatges"` i utilitzar aquest valor.
- Si escriu "Orto 196", s’ha d’inferir que és `"Orto 1965"`.

Regla clau:
- La substitució només és vàlida si el valor suggerit pertany a la llista tancada de valors vàlids. El sistema **no ha de crear ni utilitzar valors que no existeixin literalment a la llista oficial**.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALORS PERMESOS:

Valors vàlids per `idcategoria`:
"Cartografia", "CartografiaHistorica", "CartografiaMarina", "CartografiaTematica", "Geodesia", "GuiaMetropolitana", "Lidar", "Models3D", "ModelsDigitals", "ModelsDigitalsHistorics", "OrtofotoPlatges", "Ortofotos", "VolsHistorics"

Valors vàlids per `nomproducte`:
"Altres", "Batimetria multifeix", "Cartografia topobatimètrica", "Guia fulls", "Guia municipis", "Lidar 2012-2013", "Lidar 2022", "Lidar platges 2017", "Mapa Usos Sòl 1956", "Mapa Usos Sòl 1965", "Mapa Usos Sòl 1977", "Mapa Usos Sòl 1990", "Mapa Usos Sòl 2000", "Mapa Usos Sòl 2006", "Mapa Usos Sòl 2011", "Mapa Usos Sòl 2016", "Model 3D realista territori", "Model 3D territori", "Model BIM del territori", "Model d’elevacions topobatimètric", "Model elevacions", "Model Elevacions 1977", "Model ombres", "Model orientacions", "Model pendents", "Model pendents 20", "MTM 1000", "MTM 10000 analògic 1970", "MTM 2000 analògic", "MTM 2000 digital", "MTM 5000 analògic 1977", "MTM 500 analògic", "Orto 1956", "Orto 1965", "Orto 1974", "Orto 1977", "Orto 1981", "Orto 1992", "Orto 2020", "Orto Platges 2012", "Orto Platges 2013", "Orto Platges 2015", "Orto Platges 2016", "Orto Platges 2022", "Orto Platges 2023", "Orto Platges estiu 2017", "Orto Platges hivern 2017", "Vol 1956", "Vol 1961", "Vol 1965", "Vol 1970", "Vol 1972", "Vol 1974", "Vol 1977", "Vol 1979", "Vol 1981", "Vol 1982 Cerdanyola", "Vol 1983", "Vol 1985", "Vol 1987 Tiana", "Vol 1989A", "Vol 1990", "Vol 1992", "Xarxa geodèsica"
fas un
Valors vàlids per `format`:
"ASC", "COG", "DGN", "DWG", "DWGGris", "DXF", "ECW", "GIF", "GLB", "GPK", "IFC2x3", "JP2", "JPG", "KMZ", "LAZ", "OBJ", "PDF", "SHP", "SID", "SKP", "TIF", "XYZ"

Valors vàlids per `idioma`:
"Català", "Castellà", "Anglès"

Valors vàlids per `nomperfil`:
"Acadèmic", "Altres", "Privat", "Públic"

Valors vàlids per `nomambito`:
"Jurídic i financer", "Educació i farmàcia", "Cartografia i geomàtica", "Recerca i desenvolupament", "Edificació i obra civil", "Arquitectura", "Telecomunicacions", "Oci i cultura", "Agricultura - forestal - ramaderia i pesca", "Altres", "Medi ambient", "Habitatge", "Indústria i energia", "Transport i comunicacions", "Ordenació del territori i urbanisme", "Protecció civil", "Sanitat", "Comerç i turisme"

Valors vàlids per `procedencia`:
"CARTOGRAFIA", "PLANEJAMENT"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERPRETACIÓ DE CLASSIFICACIONS D'USUARIS (per perfil, àmbit o procedència):

Per a totes les consultes que fan referència a la classificació d’usuaris (per perfil, àmbit o procedència), cal distingir bé entre els diferents camps disponibles a la taula `usuarios`, ja que poden tenir significats similars però no són equivalents:

- `nomperfil`: fa referència a "públic", "privat", "acadèmic", etc.
- `nomambito`: s'utilitza per termes com “arquitectura”, “medi ambient”, “recerca”, etc.
- `procedencia`: s’utilitza només quan es fa referència explícita a l’origen institucional,  "CARTOGRAFIA" o "PLANEJAMENT".

1. **Si la pregunta inclou valors concrets com “arquitectura”, “medi ambient”, “recerca”, etc.**, s’ha d’utilitzar el camp `nomambito`.

2. **Si la pregunta fa referència a “públic”, “privat” o “acadèmic”**, s’ha d’utilitzar `nomperfil`.

3. Regles per a casos ambigus o combinats:
Si la pregunta menciona “procedència”, però el valor associat correspon a nomambito o nomperfil, s’ha de donar prioritat a nomambito o nomperfil.
Si s’indica que els usuaris venen d’un “àmbit” o “sector”, però el valor és “Públic”, “Privat” o “Acadèmic”, s’ha d’interpretar com nomperfil.
Si s’especifica que es tracta del “perfil” dels usuaris, però es proporciona un valor que coincideix amb un dels valors de nomambito, s’ha de prioritzar nomambito.

4. **Si només s’especifica "perfil", "procedència" o "àmbit" com a categoria de classificació**, interpreta-ho així:
   - "perfil" → `nomperfil`
   - "procedència" → `procedencia`
   - "àmbit" → `nomambito`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGLES PER A CONSULTES AMB DESCÀRREGUES:

Quan una pregunta implica quantificar o analitzar descàrregues segons característiques dels usuaris (com perfil, àmbit, idioma o procedència), és obligatori fer un JOIN entre descargas i usuarios utilitzant el camp usuario

Quan una pregunta implica analitzar clics (quantitat, distribució, idioma, categoria, etc.) segons informació de l’usuari, s’ha de fer JOIN entre click i usuarios mitjançant el camp usuario.

Això és essencial per relacionar els clics amb característiques com nomperfil, nomambito o procedencia.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONSULTES I REGLAMENT PER COMPONENT ESPACIAL I GENERACIÓ DE MAPES
Una consulta es considera espacial quan la pregunta de l’usuari inclou termes com “mapa”, “visualització geogràfica”, “zonificació”, “distribució territorial”, “municipis”, “districtes”, “zones”, “ubicacions”, “fulls”, “geometria”, “àrees”, “latitud”, “longitud”, “on” o expressions similars. També quan es demana localitzar un comportament com descàrregues o clics. Les regles següents estableixen com generar correctament consultes SQL amb component geoespacial.

🔸 Taula click
La taula click ja conté una columna geom (geometria de punt en EPSG:4326) generada a partir de lon i lat. Per tant:

Si es pregunta on es fan els clics, es pot fer directament un ST_Contains o ST_Intersects entre click.geom i la geometria dels municipis (divisions_administratives) o districtes (districtes_barcelona).

Si es demana un mapa de calor de clics, no cal fer operacions espacials: es poden visualitzar directament les coordenades lon i lat amb colors segons la densitat.

Per representar clics segons atributs dels usuaris, es pot fer un JOIN entre click i usuarios pel camp usuario, i visualitzar els punts amb colors segons nomperfil, nomambito, procedencia o idioma.

Si es volen diferenciar punts segons atributs propis del clic com idcategoria, idproducto o idioma, no cal cap JOIN: es poden usar directament per a aplicar color.

🔸 Taula descargas
La taula descargas conté directament la geometria (geom) i el centreide (centroid_geom) del full. Per tant:

Si es pregunta en quin municipi (nommuni) o districte (nom) es fan descàrregues o clics, no s'ha de buscar cap columna amb aquests noms a descargas o click.

En el seu lloc, s'ha de fer un JOIN espacial amb la taula corresponent:

districtes_barcelona quan es mencionin districtes.

divisions_administratives quan es mencionin municipis.

La geometria de comparació ( ST_Contains) ha de ser sempre descargas.geom o click.geom respecte a districtes_barcelona.geom o divisions_administratives.geom.

Per assegurar que una descàrrega es troba principalment dins d’un límit, es pot afegir una condició com:
ST_Area(ST_Intersection(descargas.geom, a.geom)) / ST_Area(descargas.geom) > 0.5.

Quan una consulta espacial demana identificar zones amb més descàrregues o clics però no fa referència explícita a municipis (nommuni) ni districtes (nom), s’ha d’utilitzar:

descargas.geom per representar zones de descàrregues

click.geom per representar zones de clics

No cal fer cap operació espacial (ST_Contains) en aquest cas, però és obligatori incloure geom al SELECT per poder generar correctament el mapa.

⚠️ Regla obligatòria per a consultes espacials:

Quan una consulta espacial fa referència a municipis (nommuni) o districtes (nom), la geometria (geom) de la taula divisions_administratives o districtes_barcelona respectivament s’ha d’incloure obligatòriament al SELECT final.

Si es fa servir descargas.geom o fulls.geom per representar zones, també s’ha d’incloure idfull al SELECT per garantir la traçabilitat de la geometria.

Si el model retorna només el nom (nommuni o nom) sense la geometria (geom), o retorna una geometria sense el seu idfull associat, la resposta s’ha de considerar incompleta i no apte per generar el mapa.

🔁 Evitar duplicació de `ST_Contains` en consultes espacials

Quan es realitza un `JOIN` espacial entre `descargas.geom` o `click.geom` i la geometria de `divisions_administratives` o `districtes_barcelona` mitjançant `ST_Contains(...)`, **no s'ha de tornar a repetir aquest `JOIN` en cap subconsulta posterior** si ja s'han obtingut:

- el nom administratiu (`nommuni`, `nom`)  
- i la geometria (`geom`) de la unitat espacial

Aquestes dades es poden propagar i utilitzar directament al `SELECT` final.

❌ Exemple de patró incorrecte (que duplica càlculs):
WITH primer_join AS (
    SELECT d.geom, da.nommuni, da.geom AS geom_municipi
    FROM descargas d
    JOIN divisions_administratives da ON ST_Contains(da.geom, d.geom)
)
SELECT ...
FROM primer_join
JOIN divisions_administratives da ON ST_Contains(da.geom, primer_join.geom_municipi)

✅ Patró correcte:
SELECT da.nommuni, COUNT(*) AS total_descargas, da.geom
FROM descargas d
JOIN divisions_administratives da ON ST_Contains(da.geom, d.geom)
WHERE d.nomproducte = 'Orto 2020' AND EXTRACT(YEAR FROM d.fechadescarga) = 2023
GROUP BY da.nommuni, da.geom;

🔒 Regla obligatòria:
No facis un segon `JOIN` espacial si ja tens `geom` i `nommuni` o `nom` disponibles a partir del primer `JOIN`. Si ho fas, la consulta serà ineficient i pot provocar errors de timeout (`statement timeout`).



"""    

        )

          # 3. Agregar DDL (estructura de tablas)
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )


        # 1. Primero agregar static_documentation si existe
        #if self.static_documentation != "":
            #doc_list = [static_documentation] + doc_list  # Añadimos static_documentation al principio

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



        return sql

    
    def generate_sql(self, question: str, **kwargs) -> str:
        # Usa la función base de Vanna
        sql = super().generate_sql(question, **kwargs)

        # Limpiezas comunes
        sql = sql.replace("\\_", "_")
        sql = sql.replace("\\", "")

        # Corrección de errores comunes como 'any' mal usado
        sql = self.corregir_sql_any(sql)

        # Extrae la SQL final (por si viene con bloques adicionales)
        return self.extract_sql_query(sql)


 # Si el prompt és un string pla, l'envoltem com a missatge 'user'
        #if isinstance(prompt, str):
            #prompt = [{"role": "user", "content": prompt}]

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

        print("\n==== PROMPT REAL AL MODELO ====\n", full_prompt)
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
    
