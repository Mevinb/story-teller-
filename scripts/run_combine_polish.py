import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))

import config
from pipeline.gemini_combiner import analyze_and_polish, combine_chapters

logging.basicConfig(level=logging.INFO)

def main():
    project_name = "surviving_together"
    project_dir = os.path.join(config.PROJECTS_DIR, project_name)
    chapters_dir = os.path.join(project_dir, "chapters")
    
    # Read state
    state_path = os.path.join(project_dir, "state.json")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    metadata = state.get("metadata", {})
    premise = metadata.get("premise", "")
    
    print("Combining chapters...")
    combined = combine_chapters(chapters_dir, metadata)
    print(f"Combined original length: {len(combined)} chars")
    
    # We will modify gemini_combiner.py directly to log lengths,
    # or we can print them here if we inspect the return value.
    # To do it easily, let's run analyze_and_polish and print progress.
    def progress_cb(event, data=None):
        print(f"[{event}] {data}")
        
    print("Running analyze_and_polish...")
    result = analyze_and_polish(
        combined,
        progress_cb=progress_cb,
        premise=premise,
        model_name="gemini-2.5-flash-lite"
    )
    
    print("\n--- COMBINE RESULTS ---")
    print(f"Original Chars: {len(combined)}")
    print(f"Revised Chars: {len(result['final_story'])}")
    print(f"Coverage Gaps: {result['coverage_gaps']}")
    print(f"Inserted Sections: {len(result['inserted_sections'])}")
    
    # Save the output to temporary files to check
    with open("test_combined_polished.md", "w", encoding="utf-8") as f:
        f.write(result["final_story"])
    with open("test_story_analysis.md", "w", encoding="utf-8") as f:
        f.write(result["analysis"])
        
    print("\nSaved output to test_combined_polished.md and test_story_analysis.md")

if __name__ == "__main__":
    main()
