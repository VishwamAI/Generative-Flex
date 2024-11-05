from pathlib import Path
import os
import re


def fix_indentation_issues(self
                    content): """Fix common indentation issues."""        lines = content.split("\n")
        fixed_lines = []
        indent_level = 0
        
        for line in lines: stripped = line.lstrip()        
            # Adjust indent level based on content
if stripped.startswith(("class "
                "def ")): 
        indent_level = 0 if stripped.startswith("class ") else 1
elif stripped.startswith(("if "
            "for "
            "while "
            "try: "
            "else: "
            "elif ")): 
            indent_level += 1
            elif stripped == "":                fixed_lines.append("")
                continue

                # Apply indentation
                fixed_lines.append("    " * indent_level + stripped)

                # Adjust indent level for next line
                if stripped.endswith(":") and not stripped.startswith(
("else: "
                    "elif "
                    "except: "
                    "finally: ")
                ):
                    indent_level += 1

                    return "\n".join(fixed_lines)


                def main(self):                    """Process all Python files that failed formatting."""        # List of files that failed formatting
        failed_files = [
        "src/models/multimodal/image_processor.py",
        "src/models/multimodal/base_transformer.py",
        "src/models/reasoning/math_config.py",
        "src/models/reasoning/math_head.py",
        "src/models/multimodal/multimodal_transformer.py",
        "src/models/transformer.py",
        "src/models/video_model.py",
        "src/test_simple_cot.py",
        "src/train_chatbot.py",
        "src/train_cot_fixed.py",
        "src/train_cot_simple.py",
        "src/train_minimal.py",
        "src/train_minimal_cot.py",
        "src/train_seq2seq_cot.py",
        "src/training/accelerated_trainer.py",
        "src/train_simple_cot.py",
        "src/training/train_mmmu.py",
        "src/training/jax_trainer.py",
        "src/training/trainer.py",
        "src/training/utils/timeout.py",
        "src/utils/device_config.py",
        "src/utils/environment_setup.py",
        "src/utils/training_utils.py",
        "src/models/apple_optimizations.py",
        "src/models/audio_model.py",
        "src/models/enhanced_transformer.py",
        "src/models/base_model.py",
        "src/models/generation/text2x_pipeline.py",
        "src/models/image_model.py",
        "src/models/knowledge_retrieval.py",
        "src/models/language_model.py",
        "src/models/layers/enhanced_transformer.py",
        "src/models/layers/flash_moe.py",
        ]
        
        success_count = 0
        for file_path in failed_files: ifos.path.exists(file_path) and process_file(file_path):
        success_count += 1

        print(f"\nProcessed {success_count}/{len(failed_files)} files successfully")

        # Run black formatter
        print("\nRunning black formatter...")
        os.system("python3 -m black .")


        if __name__ == "__main__":            main()