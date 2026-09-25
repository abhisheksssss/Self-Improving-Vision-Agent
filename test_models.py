"""Phase 1 verification using LangChain model providers."""
import base64
import io
from PIL import Image, ImageDraw
from langchain_core.messages import HumanMessage
from vision_agent.config import settings
from vision_agent.model import ModelFactory


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def main():
    print("=" * 60)
    print("SIVAC Phase 1: LangChain Model Layer Verification")
    print("=" * 60)
    print(f"Vision Provider:    {settings.VISION_PROVIDER} ({settings.VISION_MODEL})")
    print(f"Reasoning Provider: {settings.REASONING_PROVIDER} ({settings.REASONING_MODEL})")
    print("-" * 60)

    # Test 1: Reasoning Model (text only)
    print("\n[1/2] Testing Reasoning Model...")
    try:
        model = ModelFactory.get_reasoning_model()
        response = model.invoke([
            HumanMessage(content="You are SIVAC. Explain your role in one sentence.")
        ])
        print(f"[SUCCESS] Reasoning Model Response:\n  {response.content.strip()}")
    except Exception as e:
        print(f"[ERROR] Reasoning Model Failed: {e}")

    # Test 2: Vision Model (image + text)
    print("\n[2/2] Testing Vision Model (synthetic image)...")
    try:
        img = Image.new("RGB", (400, 200), color=(245, 245, 245))
        d = ImageDraw.Draw(img)
        d.text((40, 80), "SIVAC Vision Test - Google Search Box", fill=(0, 0, 0))

        b64 = image_to_base64(img)
        model = ModelFactory.get_vision_model()
        response = model.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "Describe the text visible in this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ])
        ])
        print(f"[SUCCESS] Vision Model Response:\n  {response.content.strip()}")
    except Exception as e:
        print(f"[ERROR] Vision Model Failed: {e}")

    print("\n" + "=" * 60)
    print("Phase 1 LangChain Verification Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
