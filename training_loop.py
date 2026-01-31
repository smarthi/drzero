"""
Real Training Loop for Dr. Zero Agent
======================================

This module implements actual LLM fine-tuning using:
1. HRPO-style data preparation (group by hop count, compute relative rewards)
2. LoRA fine-tuning via PEFT for parameter-efficient training
3. DPO (Direct Preference Optimization) for preference learning
4. Integration with Restate for durable training orchestration

The key insight from Dr. Zero:
- Group queries by structural complexity (hop count)
- Compute group-level baselines
- Use relative reward (rating - baseline) to identify positive/negative examples
- Fine-tune on positive examples (SFT) or use preference pairs (DPO)
"""

import os
import json
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import hashlib


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for the training loop"""
    # Model settings
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Small model for demo
    model_max_length: int = 2048
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # HRPO-specific settings
    min_samples_per_group: int = 10
    positive_reward_threshold: float = 0.3  # relative reward > this = positive
    negative_reward_threshold: float = -0.3  # relative reward < this = negative
    
    # Paths
    output_dir: str = "./training_output"
    checkpoint_dir: str = "./checkpoints"
    training_data_path: str = "./training_data"


@dataclass
class TrainingExample:
    """A single training example"""
    query: str
    response: str
    context_chunks: List[str]
    rating: int
    hop_count: int
    relative_reward: float
    query_id: str
    timestamp: str


@dataclass
class PreferencePair:
    """A preference pair for DPO training"""
    query: str
    context: str
    chosen_response: str
    rejected_response: str
    chosen_reward: float
    rejected_reward: float


# =============================================================================
# Data Preparation (HRPO-style)
# =============================================================================

class HRPODataPreparer:
    """
    Prepare training data using HRPO (Hop-Grouped Relative Policy Optimization).
    
    Key concepts:
    1. Group queries by hop count (structural complexity)
    2. Compute per-group baselines
    3. Calculate relative rewards
    4. Select positive examples (above baseline) for SFT
    5. Create preference pairs for DPO
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def prepare_from_feedback(
        self, 
        feedback_data: List[Dict],
        query_results: Dict[str, Dict]  # query_id -> result data
    ) -> Dict[str, any]:
        """
        Process raw feedback into HRPO-grouped training data.
        
        Args:
            feedback_data: List of feedback records with ratings
            query_results: Map of query_id to search results (query, response, chunks)
            
        Returns:
            Dictionary with grouped examples, stats, and ready-to-train data
        """
        # Group by hop count
        hop_groups: Dict[int, List[TrainingExample]] = {}
        
        for feedback in feedback_data:
            query_id = feedback.get("query_id")
            result = query_results.get(query_id)
            
            if not result:
                continue
                
            hop_count = result.get("expanded_queries", {}).get("hop_count", 1)
            
            example = TrainingExample(
                query=result.get("original_query", ""),
                response=result.get("generated_response", ""),
                context_chunks=[c.get("content", "") for c in result.get("retrieved_chunks", [])],
                rating=feedback.get("rating", 3),
                hop_count=hop_count,
                relative_reward=0.0,  # Will be computed below
                query_id=query_id,
                timestamp=feedback.get("timestamp", ""),
            )
            
            if hop_count not in hop_groups:
                hop_groups[hop_count] = []
            hop_groups[hop_count].append(example)
        
        # Compute per-group baselines and relative rewards
        group_stats = {}
        all_examples = []
        
        for hop_count, examples in hop_groups.items():
            if len(examples) < self.config.min_samples_per_group:
                continue
                
            # Compute baseline (mean rating for this hop group)
            baseline = sum(e.rating for e in examples) / len(examples)
            
            # Compute relative rewards
            for example in examples:
                example.relative_reward = example.rating - baseline
            
            group_stats[hop_count] = {
                "total_examples": len(examples),
                "baseline": round(baseline, 3),
                "positive_count": sum(1 for e in examples if e.relative_reward > self.config.positive_reward_threshold),
                "negative_count": sum(1 for e in examples if e.relative_reward < self.config.negative_reward_threshold),
            }
            
            all_examples.extend(examples)
        
        # Select training data
        positive_examples = [e for e in all_examples if e.relative_reward > self.config.positive_reward_threshold]
        negative_examples = [e for e in all_examples if e.relative_reward < self.config.negative_reward_threshold]
        
        return {
            "positive_examples": positive_examples,
            "negative_examples": negative_examples,
            "group_stats": group_stats,
            "total_processed": len(all_examples),
            "ready_for_sft": len(positive_examples) >= self.config.min_samples_per_group,
            "ready_for_dpo": len(positive_examples) >= self.config.min_samples_per_group and 
                           len(negative_examples) >= self.config.min_samples_per_group,
        }
    
    def create_sft_dataset(self, positive_examples: List[TrainingExample]) -> List[Dict]:
        """
        Create SFT (Supervised Fine-Tuning) dataset from positive examples.
        
        Format: [{"prompt": ..., "completion": ...}]
        """
        dataset = []
        
        for example in positive_examples:
            # Build context from chunks
            context = "\n\n".join(example.context_chunks[:3])
            
            # Create instruction-following format
            prompt = self._format_prompt(example.query, context)
            
            dataset.append({
                "prompt": prompt,
                "completion": example.response,
                "metadata": {
                    "query_id": example.query_id,
                    "hop_count": example.hop_count,
                    "relative_reward": example.relative_reward,
                }
            })
        
        return dataset
    
    def create_dpo_dataset(
        self, 
        positive_examples: List[TrainingExample],
        negative_examples: List[TrainingExample]
    ) -> List[PreferencePair]:
        """
        Create DPO (Direct Preference Optimization) dataset.
        
        Pairs positive and negative examples for the same query type (hop count).
        """
        pairs = []
        
        # Group by hop count
        pos_by_hop = {}
        neg_by_hop = {}
        
        for e in positive_examples:
            if e.hop_count not in pos_by_hop:
                pos_by_hop[e.hop_count] = []
            pos_by_hop[e.hop_count].append(e)
            
        for e in negative_examples:
            if e.hop_count not in neg_by_hop:
                neg_by_hop[e.hop_count] = []
            neg_by_hop[e.hop_count].append(e)
        
        # Create pairs within same hop group
        for hop_count in pos_by_hop:
            if hop_count not in neg_by_hop:
                continue
                
            pos_list = pos_by_hop[hop_count]
            neg_list = neg_by_hop[hop_count]
            
            # Pair positive with negative examples
            for pos_ex in pos_list:
                # Find a negative example with similar query (or just pick one)
                neg_ex = min(neg_list, key=lambda n: abs(len(n.query) - len(pos_ex.query)))
                
                context = "\n\n".join(pos_ex.context_chunks[:3])
                
                pair = PreferencePair(
                    query=pos_ex.query,
                    context=context,
                    chosen_response=pos_ex.response,
                    rejected_response=neg_ex.response,
                    chosen_reward=pos_ex.relative_reward,
                    rejected_reward=neg_ex.relative_reward,
                )
                pairs.append(pair)
        
        return pairs
    
    def _format_prompt(self, query: str, context: str) -> str:
        """Format prompt in instruction-following style"""
        return f"""You are a helpful search assistant. Answer the user's question based on the provided context.

<context>
{context}
</context>

<question>
{query}
</question>

Provide a clear, accurate answer based on the context. If the context doesn't contain enough information, say so."""


# =============================================================================
# LoRA Training with PEFT
# =============================================================================

class LoRATrainer:
    """
    Fine-tune LLM using LoRA (Low-Rank Adaptation) via PEFT.
    
    This enables efficient training on consumer hardware.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup(self):
        """Load base model and apply LoRA"""
        print(f"Loading base model: {self.config.base_model}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "Please install required packages: "
                "pip install transformers peft accelerate bitsandbytes"
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
        
        return self
    
    def train_sft(self, dataset: List[Dict], validation_split: float = 0.1):
        """
        Supervised fine-tuning on positive examples.
        
        Args:
            dataset: List of {"prompt": ..., "completion": ...} dicts
            validation_split: Fraction of data for validation
        """
        from transformers import TrainingArguments, Trainer
        from torch.utils.data import Dataset
        
        class SFTDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                
                # Combine prompt and completion
                full_text = item["prompt"] + "\n\n" + item["completion"]
                
                # Tokenize
                encoding = self.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                # For causal LM, labels = input_ids
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze(),
                }
        
        # Split data
        split_idx = int(len(dataset) * (1 - validation_split))
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]
        
        train_dataset = SFTDataset(train_data, self.tokenizer, self.config.model_max_length)
        val_dataset = SFTDataset(val_data, self.tokenizer, self.config.model_max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            save_total_limit=3,
            fp16=True,
            report_to=[],  # Disable wandb etc.
        )
        
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        print(f"Starting SFT training with {len(train_data)} examples...")
        trainer.train()
        
        # Save the LoRA adapter
        adapter_path = Path(self.config.output_dir) / "lora_adapter"
        self.peft_model.save_pretrained(adapter_path)
        print(f"Saved LoRA adapter to {adapter_path}")
        
        return {
            "status": "completed",
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "adapter_path": str(adapter_path),
        }
    
    def train_dpo(self, preference_pairs: List[PreferencePair]):
        """
        Direct Preference Optimization training.
        
        DPO directly optimizes the policy using preference pairs without
        requiring a separate reward model.
        """
        try:
            from trl import DPOTrainer, DPOConfig
        except ImportError:
            raise ImportError("Please install trl: pip install trl")
        
        from transformers import TrainingArguments
        from torch.utils.data import Dataset
        
        # Convert to DPO format
        dpo_data = []
        for pair in preference_pairs:
            prompt = f"{pair.context}\n\nQuestion: {pair.query}\n\nAnswer:"
            dpo_data.append({
                "prompt": prompt,
                "chosen": pair.chosen_response,
                "rejected": pair.rejected_response,
            })
        
        # Create HF Dataset
        from datasets import Dataset as HFDataset
        dpo_dataset = HFDataset.from_list(dpo_data)
        
        # DPO training config
        dpo_config = DPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size // 2,  # DPO needs more memory
            gradient_accumulation_steps=self.config.gradient_accumulation_steps * 2,
            learning_rate=self.config.learning_rate / 10,  # Lower LR for DPO
            warmup_ratio=self.config.warmup_ratio,
            beta=0.1,  # DPO temperature
            fp16=True,
            logging_steps=10,
            save_steps=100,
            report_to=[],
        )
        
        dpo_trainer = DPOTrainer(
            model=self.peft_model,
            args=dpo_config,
            train_dataset=dpo_dataset,
            tokenizer=self.tokenizer,
        )
        
        print(f"Starting DPO training with {len(dpo_data)} preference pairs...")
        dpo_trainer.train()
        
        # Save adapter
        adapter_path = Path(self.config.output_dir) / "lora_adapter_dpo"
        self.peft_model.save_pretrained(adapter_path)
        print(f"Saved DPO-trained adapter to {adapter_path}")
        
        return {
            "status": "completed",
            "preference_pairs": len(dpo_data),
            "adapter_path": str(adapter_path),
        }
    
    def load_adapter(self, adapter_path: str):
        """Load a trained LoRA adapter"""
        from peft import PeftModel
        
        if self.model is None:
            self.setup()
        
        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        print(f"Loaded adapter from {adapter_path}")
        return self
    
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response using the fine-tuned model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.peft_model.device)
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        response = response[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        
        return response.strip()


# =============================================================================
# Training Orchestrator (integrates with Restate)
# =============================================================================

class TrainingOrchestrator:
    """
    Orchestrates the full training pipeline:
    1. Fetches feedback data from Restate stores
    2. Prepares HRPO-grouped training data
    3. Runs training (SFT or DPO)
    4. Saves checkpoints and metrics
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_preparer = HRPODataPreparer(config)
        self.trainer = None
        
    async def fetch_training_data(self, restate_url: str = "http://localhost:8080") -> Dict:
        """Fetch feedback and query results from Restate stores"""
        import httpx
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get learning signals
            signals_resp = await client.post(
                f"{restate_url}/learning-store/global/get_learning_signals",
                json={},
            )
            signals = signals_resp.json() if signals_resp.status_code == 200 else {}
            
            # Get feedback history from all users
            # In production, you'd iterate over users or have a global store
            feedback_resp = await client.post(
                f"{restate_url}/feedback-store/default/get_feedback_history",
                json={},
            )
            feedback = feedback_resp.json().get("history", []) if feedback_resp.status_code == 200 else []
            
        return {
            "signals": signals,
            "feedback": feedback,
        }
    
    def prepare_training_data(
        self, 
        feedback_data: List[Dict],
        query_results: Dict[str, Dict]
    ) -> Dict:
        """Prepare HRPO-grouped training data"""
        return self.data_preparer.prepare_from_feedback(feedback_data, query_results)
    
    def run_sft_training(self, prepared_data: Dict) -> Dict:
        """Run supervised fine-tuning"""
        if not prepared_data.get("ready_for_sft"):
            return {
                "status": "insufficient_data",
                "message": f"Need at least {self.config.min_samples_per_group} positive examples",
                "current_positive": len(prepared_data.get("positive_examples", [])),
            }
        
        # Create SFT dataset
        sft_dataset = self.data_preparer.create_sft_dataset(
            prepared_data["positive_examples"]
        )
        
        # Initialize trainer and run
        self.trainer = LoRATrainer(self.config)
        self.trainer.setup()
        
        result = self.trainer.train_sft(sft_dataset)
        
        # Save training metadata
        metadata = {
            "training_type": "sft",
            "timestamp": datetime.utcnow().isoformat(),
            "config": asdict(self.config),
            "group_stats": prepared_data["group_stats"],
            "result": result,
        }
        
        metadata_path = Path(self.config.output_dir) / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return result
    
    def run_dpo_training(self, prepared_data: Dict) -> Dict:
        """Run DPO preference training"""
        if not prepared_data.get("ready_for_dpo"):
            return {
                "status": "insufficient_data",
                "message": "Need both positive and negative examples for DPO",
                "current_positive": len(prepared_data.get("positive_examples", [])),
                "current_negative": len(prepared_data.get("negative_examples", [])),
            }
        
        # Create preference pairs
        preference_pairs = self.data_preparer.create_dpo_dataset(
            prepared_data["positive_examples"],
            prepared_data["negative_examples"]
        )
        
        # Initialize trainer and run
        self.trainer = LoRATrainer(self.config)
        self.trainer.setup()
        
        result = self.trainer.train_dpo(preference_pairs)
        
        return result


# =============================================================================
# Restate Service for Training (optional - can be run standalone)
# =============================================================================

def create_training_service():
    """Create a Restate service for training orchestration"""
    try:
        import restate
        from restate import Service, Context
    except ImportError:
        print("Restate not installed, training service not available")
        return None
    
    training_service = Service("training-service")
    
    @training_service.handler()
    async def trigger_evolution(ctx: Context, config_dict: dict) -> dict:
        """Trigger a training evolution based on accumulated feedback"""
        config = TrainingConfig(**config_dict) if config_dict else TrainingConfig()
        orchestrator = TrainingOrchestrator(config)
        
        # Fetch data
        data = await orchestrator.fetch_training_data()
        
        if not data["feedback"]:
            return {"status": "no_feedback", "message": "No feedback data available"}
        
        # In a real implementation, you'd also fetch query results
        # For now, return that we need this data
        return {
            "status": "data_fetched",
            "feedback_count": len(data["feedback"]),
            "signals": data["signals"],
            "next_step": "Call prepare_and_train with query_results",
        }
    
    @training_service.handler()
    async def prepare_and_train(ctx: Context, params: dict) -> dict:
        """Prepare data and run training"""
        config = TrainingConfig(**params.get("config", {}))
        orchestrator = TrainingOrchestrator(config)
        
        feedback = params.get("feedback", [])
        query_results = params.get("query_results", {})
        training_type = params.get("training_type", "sft")
        
        # Prepare data
        prepared = await ctx.run(
            "prepare_data",
            lambda: orchestrator.prepare_training_data(feedback, query_results)
        )
        
        # Run training (this is CPU/GPU intensive)
        if training_type == "dpo":
            result = await ctx.run("train_dpo", lambda: orchestrator.run_dpo_training(prepared))
        else:
            result = await ctx.run("train_sft", lambda: orchestrator.run_sft_training(prepared))
        
        return result
    
    return training_service


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI for running training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dr. Zero Training Loop")
    parser.add_argument("--mode", choices=["demo", "sft", "dpo", "check"], default="demo",
                       help="Training mode")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Base model to fine-tune")
    parser.add_argument("--output", default="./training_output",
                       help="Output directory")
    parser.add_argument("--data", help="Path to training data JSON")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        base_model=args.model,
        output_dir=args.output,
    )
    
    if args.mode == "demo":
        # Run demo with synthetic data
        print("Running training demo with synthetic data...")
        
        # Create synthetic feedback data
        demo_feedback = [
            {"query_id": f"q{i}", "rating": r, "timestamp": datetime.utcnow().isoformat()}
            for i, r in enumerate([5, 4, 5, 4, 3, 2, 5, 4, 1, 4, 5, 3, 4, 5, 2])
        ]
        
        demo_results = {
            f"q{i}": {
                "original_query": f"What is topic {i}?",
                "generated_response": f"This is the response for topic {i}. It provides detailed information.",
                "retrieved_chunks": [{"content": f"Chunk content for topic {i}"}],
                "expanded_queries": {"hop_count": 1 + (i % 3)},
            }
            for i in range(15)
        }
        
        preparer = HRPODataPreparer(config)
        prepared = preparer.prepare_from_feedback(demo_feedback, demo_results)
        
        print("\n=== HRPO Data Preparation Results ===")
        print(f"Total processed: {prepared['total_processed']}")
        print(f"Positive examples: {len(prepared['positive_examples'])}")
        print(f"Negative examples: {len(prepared['negative_examples'])}")
        print(f"Ready for SFT: {prepared['ready_for_sft']}")
        print(f"Ready for DPO: {prepared['ready_for_dpo']}")
        print(f"\nGroup stats:")
        for hop, stats in prepared['group_stats'].items():
            print(f"  Hop {hop}: {stats}")
        
        if prepared['ready_for_sft']:
            print("\n=== Creating SFT Dataset ===")
            sft_data = preparer.create_sft_dataset(prepared['positive_examples'])
            print(f"SFT examples: {len(sft_data)}")
            print(f"Sample prompt:\n{sft_data[0]['prompt'][:500]}...")
        
    elif args.mode == "check":
        # Check if training dependencies are available
        print("Checking training dependencies...")
        
        deps = {
            "torch": False,
            "transformers": False,
            "peft": False,
            "trl": False,
            "datasets": False,
        }
        
        try:
            import torch
            deps["torch"] = True
            print(f"✓ PyTorch {torch.__version__}")
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name()}")
            else:
                print("  GPU: Not available (CPU training will be slow)")
        except ImportError:
            print("✗ PyTorch not installed")
        
        try:
            import transformers
            deps["transformers"] = True
            print(f"✓ Transformers {transformers.__version__}")
        except ImportError:
            print("✗ Transformers not installed")
        
        try:
            import peft
            deps["peft"] = True
            print(f"✓ PEFT {peft.__version__}")
        except ImportError:
            print("✗ PEFT not installed (needed for LoRA)")
        
        try:
            import trl
            deps["trl"] = True
            print(f"✓ TRL {trl.__version__}")
        except ImportError:
            print("✗ TRL not installed (needed for DPO)")
        
        try:
            import datasets
            deps["datasets"] = True
            print(f"✓ Datasets {datasets.__version__}")
        except ImportError:
            print("✗ Datasets not installed")
        
        if all(deps.values()):
            print("\n✓ All dependencies available!")
        else:
            missing = [k for k, v in deps.items() if not v]
            print(f"\nMissing: {missing}")
            print("Install with: pip install torch transformers peft trl datasets")
    
    elif args.mode in ["sft", "dpo"]:
        if not args.data:
            print("Please provide --data path to training data JSON")
            return
        
        with open(args.data) as f:
            data = json.load(f)
        
        orchestrator = TrainingOrchestrator(config)
        prepared = orchestrator.prepare_training_data(
            data.get("feedback", []),
            data.get("query_results", {})
        )
        
        if args.mode == "sft":
            result = orchestrator.run_sft_training(prepared)
        else:
            result = orchestrator.run_dpo_training(prepared)
        
        print(f"\nTraining result: {result}")


if __name__ == "__main__":
    main()
