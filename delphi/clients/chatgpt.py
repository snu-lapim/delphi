import json
from asyncio import sleep

import httpx

from ..logger import logger
from .client import Client, Response
from .types import ChatFormatRequest
import openai
from openai import OpenAI

# Preferred provider routing arguments.
# Change depending on what model you'd like to use.
PROVIDER = {"order": ["Together", "DeepInfra"]}


class ChatGPT(Client):
    def __init__(
        self,
        model: str,
        max_tokens: int = 3000,
        temperature: float = 1.0,
    ):
        super().__init__(model)

        # 인증
        #openai.api_key = api_key
        #self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI()

    def postprocess(self, response):
        msg = response.choices[0].message.content
        return Response(text=msg)

    async def generate(  # type: ignore
        self,
        prompt: ChatFormatRequest,
        max_retries: int = 1,
        **kwargs,  # type: ignore
    ) -> Response:  # type: ignore
        kwargs.pop("schema", None)
        # We have to decide if we want to do this like this or not
        # Currently only simulation uses generation kwargs.
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        #data = {
        #    "model": self.model,
        #    "messages": prompt,
        #    # "provider": PROVIDER,
        #    "temperature": temperature,
        #}
        #import pdb; pdb.set_trace()
        if "gpt-4o-mini" in self.model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                max_completion_tokens=max_tokens,
                reasoning_effort="medium"
            )

        result = self.postprocess(response)

        return result