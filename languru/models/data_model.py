from typing import Dict, List, Text, Type, TypeVar

import pyjson5
from openai import OpenAI
from pyassorted.string import extract_code_blocks
from pydantic import BaseModel
from pydantic_core import ValidationError

from languru.config import logger
from languru.prompts import PromptTemplate
from languru.prompts.repositories.data_model import prompt_date_model_from_openai
from languru.utils.common import display_messages, ensure_list
from languru.utils.openai_utils import ensure_openai_chat_completion_content

DataModelTypeVar = TypeVar("DataModelTypeVar", bound="DataModel")


class DataModel(BaseModel):
    @classmethod
    def model_from_openai(
        cls: Type[DataModelTypeVar],
        content: Text,
        client: "OpenAI",
        model: Text = "gpt-4o-mini",
        *,
        verbose: bool = False,
        **kwargs,
    ) -> List[DataModelTypeVar]:
        # Get schema
        schema = cls.model_json_schema()
        model_schema = {cls.__name__: schema}

        # Prepare prompt
        user_message = {
            "role": "user",
            "content": (
                "<model_json_schema>\n{model_schema}\n</model_json_schema>\n\n"
                + "{user_says}"
            ),
        }
        prompt_template = PromptTemplate(
            prompt_date_model_from_openai, messages=[user_message]
        )
        input_messages = prompt_template.format_messages(
            prompt_vars={"model_schema": model_schema, "user_says": content}
        )
        if verbose:
            display_messages(
                messages=input_messages,
                table_title=f"{client.__class__.__name__} Chat Messages Input",
            )

        # Generate response
        chat_res = client.chat.completions.create(
            messages=input_messages, model=model, temperature=0.0
        )
        chat_answer = ensure_openai_chat_completion_content(chat_res)
        if verbose:
            display_messages(
                messages=[{"role": "assistant", "content": chat_answer}],
                table_title=f"{client.__class__.__name__}({model}) Chat Response",
            )

        # Parse response
        code_blocks = extract_code_blocks(chat_answer, language="json")
        if len(code_blocks) == 0:
            raise ValueError(
                f"Failed to extract a JSON code block from the response: {chat_answer}"
            )
        code_block = code_blocks[0]  # Only one code block is expected
        items: List[Dict] = ensure_list(pyjson5.loads(code_block))

        # Validate models
        models: List[DataModelTypeVar] = []
        for item in items:
            try:
                _model = cls.model_validate(item)
            except ValidationError as e:
                raise ValueError(
                    f"Failed to validate model '{cls.__name__}' from data: {item}"
                ) from e
            models.append(_model)

        logger.debug(f"Generated models from OpenAI: {models}")
        return models
