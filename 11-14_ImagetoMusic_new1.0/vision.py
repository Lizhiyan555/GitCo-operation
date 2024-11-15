from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
import base64
import json

image = "image.png"
model = ChatOllama(model="llama3.2-vision", base_url="http://thor:11434", format="json")

# Read the image
with open(image, "rb") as f:
    image_data = f.read()
    image_data = base64.b64encode(image_data).decode("utf-8")

    # Create a message with the image
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Given this image, can you give me some mood keywords for a song to generate. Output json as {tags: [list of tags]}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ],
    )

    # Invoke the model with the message
    response = model.invoke([message])
    print(response)
    jsonresponse = json.loads(response.content)

    print(jsonresponse['tags'])
