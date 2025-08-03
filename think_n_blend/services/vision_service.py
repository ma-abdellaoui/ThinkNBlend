import os
import json
from openai import OpenAI
from think_n_blend.config import GPT4_VISION_PROMPT, GPT4_VISION_MODEL
from think_n_blend.schemas import Gpt4VisionResponse, ReferenceObject, TargetObject
from think_n_blend.utils.image_utils import encode_image

def get_vision_reasoning(main_image_path: str, object_crop_path: str) -> Gpt4VisionResponse:
    """
    Analyzes the main image and object crop to determine a realistic placement for the object.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    main_image_b64 = encode_image(main_image_path)
    object_crop_b64 = encode_image(object_crop_path)

    response = client.chat.completions.create(
        model=GPT4_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GPT4_VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{main_image_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{object_crop_b64}"}},
                ],
            }
        ],
        max_tokens=500,
    )

    response_text = response.choices[0].message.content

    try:
        json_response_string = response_text.split('```json')[1].split('```')[0].strip()
        data = json.loads(json_response_string)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing JSON from response: {e}")
        print(f"Raw response: {response_text}")
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            print("Fallback JSON parsing failed. Raising exception.")
            raise ValueError("Invalid JSON response from GPT-4 Vision") from e

    with open('output/description.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    return Gpt4VisionResponse(
        reference_object=ReferenceObject(**data["reference_object"]),
        target_object=TargetObject(**data["target_object"]),
    )
