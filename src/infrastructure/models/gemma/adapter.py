import base64
import json
from typing import Dict, Any, Optional

import requests
from src.core.config import CONFIG
from src.domain.models import OcrInput, OcrOutput
from src.domain.ports import OcrPort
from src.infrastructure.models.registry import register_adapter
from src.infrastructure.models.gemma.config import gemma_settings


@register_adapter("gemma")
class GemmaAdapter(OcrPort):
    def __init__(self):
        # Build base API URL and load the prompt only once
        self.api_url = gemma_settings.get_full_api_url(CONFIG.lms_api)
        self._load_prompt()

    def _load_prompt(self) -> None:
        try:
            with open(gemma_settings.prompt_path, "r", encoding="utf-8") as f:
                self.instruction_text = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"Prompt file not found at {gemma_settings.prompt_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading prompt file: {str(e)}")

    def _get_generation_params(self, 
                               overrides: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
        required_keys = [
            "model_name",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repeat_penalty",
            # TODO: Add max_tokens, stop_sequences, etc.
        ]

        missing = [key for key in required_keys if not hasattr(gemma_settings, key)]
        if missing:
            raise RuntimeError(f"Missing generation settings in gemma_settings: {missing}")

        # Base “generation” dict; these keys match what Gemma’s API expects.
        gen_params: Dict[str, Any] = {
            "model": gemma_settings.model_name,
            "temperature": gemma_settings.temperature,
            "top_k": gemma_settings.top_k,
            "top_p": gemma_settings.top_p,
            "min_p": gemma_settings.min_p,
            "repeat_penalty": gemma_settings.repeat_penalty,
        }

        # Allow any runtime overrides: e.g. apdapter.predict(..., overrides={"temperautre": 0.3})
        if overrides:
            for k, v in overrides.items():
                if k not in gen_params:
                    raise RuntimeError(f"Unknown hyperparameter override: {k}")
                gen_params[k] = v

        return gen_params

    def _prepare_payload(
        self, 
        image_bytes: bytes, 
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build the full JSON payload, including both the 'messages' section
        and the generation specific hyperparamets
        """
        # convert image bytes → Data URI based 64 image decoding
        img_b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{img_b64}"

        #Core messages array (instruction + image)
        messages_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.instruction_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    },
                ],
            }
        ]

        #get generation hyperparameters
        generation_params = self._get_generation_params(overrides) # merge any overrides

        # merge
        payload: Dict[str, Any] = {
            "messages": messages_payload,
            **generation_params
        }

        return payload

    def _process_response(self, response_text: str) -> Dict[str, Any]:
        """
        Strip any markdown/json special chars if requested, then parse JSON.
        Raises RuntimeError if parsing fails.
        """
        if gemma_settings.strip_json_markers:
            if response_text.startswith("```json"):
                response_text = response_text[len("```json\n") :]
            if response_text.endswith("\n```"):
                response_text = response_text[: -len("\n```")]

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse model response as JSON: {str(e)}")

    def predict(
        self, 
        ocrInput: OcrInput, 
        overrides: Optional[Dict[str, Any]] = None # For runtime generate hyperparams override
    ) -> OcrOutput:
        #build the payload
        payload = self._prepare_payload(bytes(ocrInput.bytes), overrides)

        # request Gemma API
        try:
            response = requests.post(
                self.api_url,
                headers=gemma_settings.headers,
                json=payload
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")

        # Extract the raw string content
        data = response.json()
        model_output = data["choices"][0]["message"]["content"]

        # parse JSON
        processed_output = self._process_response(model_output)

        # pewpare OcrOutput
        return OcrOutput(
            texts=processed_output["texts"],
            description={
                "description": processed_output["description"],
                "sentence": processed_output["sentence"],
            },
        )
