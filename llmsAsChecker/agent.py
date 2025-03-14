import json

with open('apikey.json', 'r') as file:
    keys = json.load(file)


from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# Define output structure using Pydantic
class NewsVerification(BaseModel):
    label: Literal["true", "fake", "NA"] = Field(description="Verdict of news authenticity")
    reason: str = Field(description="Reason for the label")

# Set up output parser
parser = PydanticOutputParser(pydantic_object=NewsVerification)

# Create prompt template
prompt = PromptTemplate(
    template="""
            You are a fact-checking assistant.
            
            Given the following news details:
            Title: {title}
            Text: {text}
            Date: {date}
            
            Determine if the news article appears more likely to be true, fake, or "NA" if policy limits your answer. 
            
            Ensure that the output is a valid ptyhon dictionary.

            {format_instructions}
            """,
    input_variables=["title", "text", "date"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Initialize LLM 
llm_openai = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.0, openai_api_key = keys['openai'])
llm_deepseek = ChatDeepSeek(model='deepseek-reasoner', temperature=0.0, api_key=keys['deepseek'])
llm_llama = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.0, api_key=keys['groq'])




def analyze_news(title: str, text: str, date: str, model:str) -> dict:
    """Process news article and return structured verification result"""
    assert model in ['openai', 'deepseek', 'llama'], 'Model not supported'
    if model == 'openai':
        verification_chain = prompt | llm_openai | parser
    elif model == 'deepseek':
        verification_chain = prompt | llm_deepseek | parser
    else:
        verification_chain = prompt | llm_llama | parser
    try:
        result = verification_chain.invoke({
            "title": title,
            "text": text,  
            "date": date
        })
        return {
            "label": result.label,
            "reason": result.reason
        }
    except Exception as e:
        return {
            "label": "Err",
            "reason": f"Analysis failed: {str(e)}"
        }

if __name__ == "__main__":

    #Test

    import pprint
    input = {
        'title': 'Scientists Invent Faster-than-Light Engine',
        'text': 'NASA declared that their scientists designed and tested the first '
        'faster-than-light engine in the world, which reached speeds up to three '
        'times the speed of light.',
        'date': 'March 10, 2024',
        'model': 'llama'
    }
    pprint.pprint(analyze_news(**input))