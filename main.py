import argparse
import json
from src.agent.prompt_agent import AutoPromptAgent
from src.utils.config import TASK_DESCRIPTION

def main():
    parser = argparse.ArgumentParser(description="Auto Prompt Agent for Image Generation")
    parser.add_argument("--mode", type=str, choices=["loop", "batch"], default="loop", 
                        help="Optimization mode: 'loop' for iterative refinement, 'batch' for variations.")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for loop mode.")
    parser.add_argument("--variations", type=int, default=5, help="Number of variations for batch mode.")
    parser.add_argument("--task", type=str, default=TASK_DESCRIPTION, help="Custom task description.")
    parser.add_argument("--model", type=str, default="qwen3.6-plus", help="Qwen model for prompt generation.")
    parser.add_argument("--gen_model", type=str, default="qwen-image-2.0-pro", help="Image generation/editing model.")
    parser.add_argument("--image", type=str, default=None, help="Initial image to start from (optional).")
    
    args = parser.parse_args()
    
    agent = AutoPromptAgent(task_description=args.task, max_iterations=args.iterations)
    agent.qwen.model = args.model
    agent.image_gen.model = args.gen_model
    
    if args.mode == "loop":
        print("Running in Loop Mode...")
        results = agent.run_optimization_loop(initial_image=args.image)
    else:
        print("Running in Batch Mode...")
        results = agent.run_batch_variations(num_variations=args.variations)
        
    # 保存结果到文件
    output_file = "outputs/results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\nTask completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
