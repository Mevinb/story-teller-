import os
import sys
import json
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

import config
from pipeline.orchestrator import PipelineOrchestrator, normalize_project_name
from pipeline.gemini_combiner import combine_chapters, analyze_and_polish

# Five story ideas defined by the user
STORIES = [
    {
        "title": "The Last Train Home",
        "genre": "survival drama",
        "premise": "A young woman named Priya misses the last train home during a heavy monsoon. She teams up with a quiet stranger named Arjun to find an alternate way back. Along the journey, they face flooded roads, broken bridges, and dangerous situations. They slowly become close friends and help each other overcome their personal traumas. The story must cover: 1. Priya realizes she missed the last train and panic sets in. 2. Priya meets Arjun under a leaking station shelter, and they agree to walk together. 3. They encounter a flooded main road and decide to detour through an unfamiliar alley. 4. Arjun helps Priya cross a rushing stream where a small footbridge is starting to collapse. 5. They find shelter in a half-built shelter and share stories about their past traumas. 6. They finally reach the city outskirts at dawn as the rain stops, parting as close friends.",
        "themes": ["survival", "trauma", "friendship"],
        "setting": "Heavy monsoon, flooded roads, broken bridges, and dark alleyways.",
        "characters": {
            "Priya": {
                "role": "main",
                "description": "An anxious but resilient young woman working in the city. Desperate to return home to her family.",
                "traits": ["anxious", "resilient", "determined"]
            },
            "Arjun": {
                "role": "main",
                "description": "A quiet, calm, and resourceful stranger carrying personal trauma. Very practical under pressure.",
                "traits": ["quiet", "resourceful", "calm", "traumatized"]
            }
        }
    },
    {
        "title": "The Forgotten Village",
        "genre": "mystery drama",
        "premise": "A city girl named Meera inherits an old house in a remote mountain village. When she visits to sell it, she discovers the villagers are hiding a mysterious secret. With the help of a local teacher named Karan, she uncovers the truth about her family’s past while learning the value of community and roots. The story must cover: 1. Meera arrives at the village and sees the neglected house. 2. The local village elder acts suspiciously and warns Meera to sell quickly and leave. 3. Meera meets Karan, the village teacher, who offers to show her around. 4. Meera finds old letters in the attic hinting at a family secret. 5. Karan helps her translate the local dialect in the letters. 6. Meera uncovers the family history and decides to stay and restore the house.",
        "themes": ["mystery", "family past", "community", "identity"],
        "setting": "A remote mountain village with old houses, mist, and dense forests.",
        "characters": {
            "Meera": {
                "role": "main",
                "description": "A city girl who inherits an old house in a mountain village. Curious, independent, but disconnected from her roots.",
                "traits": ["curious", "independent", "modern", "stubborn"]
            },
            "Karan": {
                "role": "main",
                "description": "A kind and highly educated school teacher in the remote mountain village. Warm and grounded.",
                "traits": ["kind", "grounded", "knowledgeable", "patient"]
            }
        }
    },
    {
        "title": "One Week to Live",
        "genre": "drama romance",
        "premise": "A 22-year-old college student named Riya is told she has only one week to live due to a rare illness. She decides to fulfill her bucket list with her best friend and a kind doctor. Along the way, she learns what truly matters in life and finds unexpected love. The story must cover: 1. Riya gets the diagnosis of her rare illness from Doctor Rohan. 2. Riya decides to write a bucket list instead of staying in the hospital. 3. Her best friend Diya joins her in completing the list's adventures. 4. Doctor Rohan joins them out of concern and becomes closer to Riya. 5. Riya realizes she has fallen in love with Doctor Rohan during a sunset on a hilltop. 6. Riya learns to cherish life as the week comes to a close.",
        "themes": ["mortality", "love", "bucket list", "friendship"],
        "setting": "A vibrant college town, hospital rooms, and beautiful natural outdoor landscapes.",
        "characters": {
            "Riya": {
                "role": "main",
                "description": "A 22-year-old college student. Diagnosed with a terminal illness, she is free-spirited and determined.",
                "traits": ["free-spirited", "determined", "emotional", "brave"]
            },
            "Doctor Rohan": {
                "role": "main",
                "description": "A kind, compassionate, and dedicated young doctor. Gentle and deeply caring.",
                "traits": ["compassionate", "dedicated", "gentle", "professional"]
            },
            "Diya": {
                "role": "supporting",
                "description": "Riya's supportive best friend. Extremely loyal and emotional.",
                "traits": ["loyal", "emotional", "fun-loving"]
            }
        }
    },
    {
        "title": "The Swap",
        "genre": "drama comedy",
        "premise": "Two best friends, a rich girl named Ananya and a poor but talented girl named Sneha, magically swap lives for one month. They experience each other’s struggles and privileges, leading to personal growth, stronger friendship, and important life lessons. The story must cover: 1. Ananya and Sneha complain about their respective lives. 2. A mysterious magical event swaps their bodies/lives. 3. Ananya struggles with Sneha's chores and part-time job. 4. Sneha struggles with the high expectations and coldness of high society. 5. They meet to share advice and comfort each other. 6. The swap reverts after they learn the value of each other's lives.",
        "themes": ["empathy", "friendship", "class difference", "growth"],
        "setting": "A bustling city split between high-end penthouses and humble suburban apartments.",
        "characters": {
            "Ananya": {
                "role": "main",
                "description": "A wealthy, sheltered college student. Used to luxury but lacks real-world perspective.",
                "traits": ["sheltered", "privileged", "sensitive", "kind"]
            },
            "Sneha": {
                "role": "main",
                "description": "A poor but talented student working multiple jobs. Hardworking but feels overwhelmed.",
                "traits": ["hardworking", "talented", "weary", "loyal"]
            }
        }
    },
    {
        "title": "Echoes of the Past",
        "genre": "adventure romance",
        "premise": "A young archaeologist named Tara discovers an ancient artifact that lets her see visions of the past. While excavating a lost temple, she teams up with a skeptical historian named Vikram. Together they uncover a forgotten love story that mirrors their own growing feelings. The story must cover: 1. Tara finds the glowing ancient artifact in a hidden alcove. 2. Vikram warns her not to touch it, expressing skepticism. 3. Tara touches it and experiences a vivid vision of ancient lovers in the temple. 4. She describes the vision to Vikram, who starts investigating the temple inscriptions. 5. They find clues about the lovers' tragic fate as they solve temple puzzles together. 6. They realize their own feelings mirror the ancient love story.",
        "themes": ["archaeology", "visions", "romance", "history"],
        "setting": "A lost ancient temple excavation site deep in a tropical forest.",
        "characters": {
            "Tara": {
                "role": "main",
                "description": "A passionate and intuitive young archaeologist. Deeply connected to the past.",
                "traits": ["passionate", "intuitive", "brave", "expressive"]
            },
            "Vikram": {
                "role": "main",
                "description": "A skeptical and analytical historian. Grounded and values hard evidence.",
                "traits": ["skeptical", "analytical", "grounded", "protective"]
            }
        }
    }
]

def run_test_pipeline():
    print("==================================================")
    print("STARTING COMPREHENSIVE STORY GENERATION PIPELINE TESTS")
    print("==================================================")

    for i, s in enumerate(STORIES):
        title = s["title"]
        slug = normalize_project_name(title)
        print(f"\n[{i+1}/{len(STORIES)}] PROJECT: {title} (slug: {slug})")
        
        # 1. Initialize project
        print("  → Creating project...")
        orchestrator = PipelineOrchestrator(
            project_name=slug,
            backend="groq",
            gemini_model="gemini-2.5-flash"
        )
        
        # Delete existing project dir if it exists to ensure a clean run
        import shutil
        shutil.rmtree(orchestrator.project_dir, ignore_errors=True)
        os.makedirs(orchestrator.project_dir, exist_ok=True)
        os.makedirs(orchestrator.chapters_dir, exist_ok=True)
        os.makedirs(orchestrator.logs_dir, exist_ok=True)

        state = orchestrator.create_project(
            title=title,
            genre=s["genre"],
            premise=s["premise"],
            characters=s["characters"],
            themes=s["themes"],
            setting=s["setting"]
        )
        print(f"    ✓ Project initialized. Title: '{state['metadata']['title']}'")
        
        # Load project to set writer and editor models/settings
        orchestrator.load_project()
        
        # 2. Generate Chapter 1
        print("  → Generating Chapter 1...")
        t0 = time.time()
        c1_res = orchestrator.generate_chapter(pacing="moderate")
        t_c1 = time.time() - t0
        
        if c1_res.get("status") == "cancelled":
            print("    ✗ Chapter 1 generation was cancelled.")
            continue
            
        print(f"    ✓ Chapter 1 generated in {t_c1:.1f}s.")
        print(f"      Title: {c1_res.get('chapter_title')}")
        print(f"      Words: {c1_res.get('total_words')} words across {c1_res.get('scenes_count')} scenes.")
        print(f"      File: {c1_res.get('file')}")
        
        # 3. Read Chapter 1 (testing read function)
        print("  → Reading Chapter 1 content...")
        ch1_text = orchestrator.read_chapter(1)
        if ch1_text:
            print(f"    ✓ Successfully read Chapter 1 ({len(ch1_text)} chars). Snippet:")
            lines = ch1_text.strip().splitlines()
            snippet = "\n".join(lines[:8])
            print(f"      \"\"\"\n{snippet}\n      ...\"\"\"")
        else:
            print("    ✗ Error reading Chapter 1!")
            
        # 4. Generate Chapter 2 (to test state transitions, multi-chapter generation, and then prune)
        print("  → Generating Chapter 2...")
        t0 = time.time()
        c2_res = orchestrator.generate_chapter(pacing="moderate")
        t_c2 = time.time() - t0
        print(f"    ✓ Chapter 2 generated in {t_c2:.1f}s.")
        print(f"      Title: {c2_res.get('chapter_title')}")
        print(f"      Words: {c2_res.get('total_words')} words.")
        
        # Get status/info
        info = orchestrator.get_project_info()
        print(f"    ✓ Current project info: Chapters written = {info['chapters_written']}, Total scenes = {info['total_scenes']}")

        # 5. Delete Chapter 2 (testing deletion and state/memory rewinding)
        print("  → Testing Deletion/Pruning (rewinding Chapter 2)...")
        del_res = orchestrator.delete_chapters_from(2)
        print(f"    ✓ Pruning complete. Remaining chapters = {del_res.get('remaining_chapter_count')}, current state chapter = {del_res.get('current_chapter')}")
        
        # Verify Chapter 2 file is gone and current chapter is 1
        ch2_path = os.path.join(orchestrator.chapters_dir, "chapter_002.md")
        if not os.path.exists(ch2_path) and del_res.get("current_chapter") == 1:
            print("    ✓ Successfully deleted Chapter 2 file and updated state pointer to 1.")
        else:
            print(f"    ✗ Failed deletion verification. File exists: {os.path.exists(ch2_path)}, State: {del_res.get('current_chapter')}")

        # Let's generate Chapter 2 again so we have a multi-chapter story to test Combine & Polish
        print("  → Re-generating Chapter 2 for Combine & Polish...")
        orchestrator.generate_chapter(pacing="moderate")

        # 6. Combine & Polish
        print("  → Combining chapters and Polishing via Gemini...")
        metadata = orchestrator.state_manager.load()["metadata"]
        combined = combine_chapters(orchestrator.chapters_dir, metadata)
        print(f"    ✓ Combined length: {len(combined)} characters.")
        
        def progress_cb(event, data=None):
            if event == "combine_status":
                print(f"      [Progress] {data.get('step')}")

        t0 = time.time()
        polish_res = analyze_and_polish(
            combined,
            progress_cb=progress_cb,
            premise=s["premise"],
            model_name="gemini-2.5-flash",
            state=orchestrator.state_manager.load()
        )
        t_polish = time.time() - t0
        print(f"    ✓ Polish complete in {t_polish:.1f}s.")
        print(f"      Polished story length: {len(polish_res['final_story'])} characters.")
        print(f"      Coverage gaps: {polish_res['coverage_gaps']}")
        
        # Save output files to project directory (simulating UI/app behavior)
        original_path = os.path.join(orchestrator.project_dir, "combined_original.md")
        polished_path = os.path.join(orchestrator.project_dir, "combined_polished.md")
        analysis_path = os.path.join(orchestrator.project_dir, "story_analysis.md")
        
        with open(original_path, "w", encoding="utf-8") as f:
            f.write(combined)
        with open(polished_path, "w", encoding="utf-8") as f:
            f.write(polish_res["final_story"])
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(polish_res["analysis"])
        print(f"    ✓ Combined & polished files saved successfully.")

    print("\n==================================================")
    print("ALL TESTS RUN COMPLETED")
    print("==================================================")

if __name__ == "__main__":
    try:
        run_test_pipeline()
    except Exception as e:
        print(f"\n✗ PIPELINE CRASHED: {e}")
        traceback.print_exc()
