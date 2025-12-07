from openai import OpenAI
import os
from dotenv import load_dotenv
import instructor
import logging

log = logging.getLogger(__name__)

load_dotenv()  # reads variables from a .env file and sets them in os.environ


openAI_endpoint=os.getenv("OPENAI_URL")
openAI_apikey=os.getenv("OPENAI_APIKEY")

default_modelid="google/gemma-3-27b-it:free"

client = instructor.from_openai( 
    OpenAI(
        base_url=openAI_endpoint,
        api_key=openAI_apikey,
    ), 
    mode=instructor.Mode.JSON 
)

from pydantic import Field, BaseModel

class StringResponse(BaseModel):
    response_text: str = Field(description="response")

class LLMParameter(BaseModel):
   temperature: float = 0.0
   max_tokens: int = 4096
   max_retries: int = 5


def call(system_prompt: str, message: str, response_type:BaseModel =StringResponse, model: str = default_modelid, history: list[dict] = [], params: LLMParameter = LLMParameter() ) -> str: # type: ignore

    log.debug("call model %s with parameters: %s", model, params)

    history.append(
          {
              "role": "user",
              "content": message,
          },)

    # Gemma-3 don't support system role so we are sending system prompt as part of user messages
    result_obj = client.chat.completions.create (
      model=model,
      response_model=response_type,
      messages=[
          {"role": "user", "content": system_prompt},
      ]+history,
      temperature = params.temperature,
      max_tokens = params.max_tokens,
      max_retries= params.max_retries,
    ) 
    return result_obj


