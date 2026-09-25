"""Phase 2 verification script: tests screen capture and hybrid perception."""
from vision_agent.perception import HybridPerceptionEngine


def main():
    print("=" * 60)
    print("SIVAC Phase 2: Hybrid Perception Engine Test")
    print("=" * 60)

    # Initialize the hybrid perception orchestrator
    engine = HybridPerceptionEngine()

    print("\nCapturing screen and running perception sources...")
    print("(Running VLM + Windows UI Automation + DOM...)")

    # Run perception on the active screen (use_ocr=False by default if tesseract binary isn't in PATH)
    state = engine.perceive(use_ocr=False)

    print("\n--- Perception Results ---")
    print(f"Application   : {state.application}")
    print(f"Window Title  : {state.window_title}")
    print(f"URL           : {state.url}")
    print(f"Screenshot    : {state.screenshot_path}")
    print(f"Dimensions    : {state.dimensions[0]}x{state.dimensions[1]}")
    print(f"Total Elements: {len(state.elements)}")

    print("\n--- Detected Elements (First 15) ---")
    for elem in state.elements[:15]:
        center = elem.bbox.center
        print(f"  [{elem.source.upper():4}] {elem.id:10} | {elem.type:14} | click_at=({center[0]:4},{center[1]:4}) | text='{elem.text[:30]}'")

    print("\n" + "=" * 60)
    print("Phase 2 Perception Verification Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
