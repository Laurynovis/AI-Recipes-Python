from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="API de Receitas com IA")

# ATENÇÃO: Cole sua chave aqui dentro das aspas! 
# (Em um projeto real, depois passaremos isso para um arquivo .env oculto)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("A variável de ambiente GROQ_API_KEY não foi encontrada!")

# Inicializamos o cliente da OpenAI, mas apontando para os servidores do Groq!
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# O que o Java vai enviar
class MensagemUsuario(BaseModel):
    mensagem: str = Field(..., description="O pedido do usuário")

# O formato exato que vamos devolver
class ReceitaGerada(BaseModel):
    titulo: str
    descricao: str
    tempo_preparo_minutos: int
    ingredientes: List[str]
    passos: List[str]

@app.post("/api/gerar-receita", response_model=ReceitaGerada)
async def gerar_receita(request: MensagemUsuario):
    try:
        # 1. O Prompt de Sistema: A regra do jogo para a IA
        prompt_sistema = """
        Você é um chef de cozinha especialista. 
        Crie uma receita baseada no pedido do usuário.
        Você DEVE retornar a resposta EXATAMENTE no seguinte formato JSON, sem nenhum texto adicional antes ou depois:
        {
            "titulo": "Nome da receita",
            "descricao": "Breve descrição da receita",
            "tempo_preparo_minutos": 30,
            "ingredientes": ["ingrediente 1", "ingrediente 2"],
            "passos": ["passo 1", "passo 2"]
        }
        """

        resposta = client.chat.completions.create(
            model="llama-3.1-8b-instant", # <--- BASTA MUDAR O NOME AQUI
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": request.mensagem}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.7
        )

        # 3. Transformando a resposta (texto) em um objeto Python real
        conteudo_texto = resposta.choices[0].message.content
        conteudo_json = json.loads(conteudo_texto)
        
        # 4. O Pydantic valida se a IA não "esquecou" nenhum campo e empacota
        return ReceitaGerada(**conteudo_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))