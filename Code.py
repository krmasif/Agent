import json
import logging
import os
from datetime import datetime
from time import sleep
from typing import Dict, Any
from jsonschema import validate, ValidationError

# ADK Imports
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner, PersistentRunner
from google.adk.storage import InFileSystemStorage

# **MODIFIED**: Added Gemini imports
import google.generativeai as genai
from google.generativeai.types import GenerativeModel
from google.generativeai.types.generation_types import GenerateContentResponse

# --- Configuration ---
logging.basicConfig(level=logging.INFO, filename='agent_system.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
STORAGE_DIR = "agent_storage"


# --- Utility Functions ---
def log_event(agent_name: str, event: str, data: Dict = None):
    """Logs a structured event for an agent."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "agent": agent_name,
        "event": event,
        "data": data or {}
    }
    logging.info(json.dumps(log_entry))

# **NEW**: Helper to parse LLM JSON responses safely
def parse_llm_json(response: GenerateContentResponse) -> Dict[str, Any]:
    """Safely parses JSON from an LLM response."""
    try:
        # Extract text and strip markdown backticks
        text = response.text
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        logging.error(f"Failed to parse LLM JSON response: {e}. Response text: {response.text}")
        # Return a default error structure
        return {"error": "Failed to parse LLM response", "raw_text": response.text}


# --- Schemas ---
# Schema to validate the input for Agent 1
agent1_input_schema = {
    "type": "object",
    "properties": {
        "attemptid": {"type": "string"},
        "code_submission": {"type": "string"},
        "language": {"type": "string"},
        "problem_metadata": {
            "type": "object",
            "properties": {
                "problem_id": {"type": "string"},
                "difficulty": {"type": "string"},
                "test_cases": {"type": "array"}
            },
            "required": ["problem_id"]
        },
        "time_taken": {"type": "number"},
        "hints_requested": {"type": "number"},
        "explanation": {"type": "string"},
        "userconfidence": {"type": "string"}
    },
    "required": ["attemptid", "code_submission", "language", "problem_metadata", "time_taken", "hints_requested"]
}


# --- Agent Definitions ---

# Agent 1: User Interaction Analyzer (Stateful, uses LLM)
class UserInteractionAnalyzer(Agent):
    """Analyzes user code and interaction data using an LLM."""
    def __init__(self, llm: GenerativeModel): # **MODIFIED**: Added llm
        super().__init__(name="Agent1_UserInteractionAnalyzer",
                         description="Analyzes user code submissions for DSA learning.")
        self.llm = llm # **MODIFIED**: Store LLM client

    def run(self, inputs, context=None):
        try:
            # Validate input against the schema
            validate(instance=inputs, schema=agent1_input_schema)
            log_event(self.name, "start", {"attemptid": inputs.get("attemptid")})

            code = inputs["code_submission"]
            language = inputs["language"]
            metadata = inputs["problem_metadata"]
            explanation = inputs.get("explanation", "")

            # **MODIFIED**: Replaced mocks with actual LLM call
            prompt = f"""
            Analyze the following code submission for the DSA problem: "{metadata.get('problem_id')}".
            The user's submitted code in {language} is:
            ```
            {code}
            ```
            The user's own explanation of their approach is: "{explanation}"

            Please evaluate the code and provide a JSON object with the following keys:
            - "correctness": A string, one of "correct", "partial", or "incorrect".
            - "approachused": A brief string describing the algorithm (e.g., "brute_force", "hash_map", "two_pointers").
            - "conceptsdemonstrated": A list of key data structures or concepts used (e.g., ["arrays", "loops"]).
            - "errorsfound": A list of any syntax or logic errors found. Empty list if none.
            - "reasoningpattern": A string describing the user's reasoning (e.g., "algorithmic", "heuristic").
            
            Respond ONLY with the raw JSON object.
            """
            
            # Call the generative model
            response = self.llm.generate_content(prompt)
            analysis_data = parse_llm_json(response)

            if "error" in analysis_data:
                raise Exception(f"LLM parsing failed: {analysis_data.get('raw_text')}")

            # Structure the output for the next agent
            output = {
                "attemptid": inputs["attemptid"],
                "correctness": analysis_data.get("correctness", "incorrect"),
                "approachused": analysis_data.get("approachused", "unknown"),
                "conceptsdemonstrated": analysis_data.get("conceptsdemonstrated", []),
                "errorsfound": analysis_data.get("errorsfound", []),
                "timetakenminutes": inputs["time_taken"],
                "hintsrequested": inputs["hints_requested"],
                "userconfidence": inputs.get("userconfidence", "medium"),
                "reasoningpattern": analysis_data.get("reasoningpattern", "unknown"),
            }
            log_event(self.name, "complete", output)
            return output
        except (ValidationError, Exception) as e:
            log_event(self.name, "error", {"error": str(e)})
            raise


# Agent 2: Learning Profile Manager (Stateful, no LLM)
class LearningProfileManager(Agent):
    """Updates and tracks user learning profile and mastery persistently."""
    def __init__(self):
        super().__init__(name="Agent2_LearningProfileManager",
                         description="Updates and tracks user learning profile and mastery.")

    def run(self, inputs, context=None):
        try:
            analysis = inputs.get("analysis")
            # Load the user's profile from persistent state
            profile = context.state.get("profile", {}) 

            log_event(self.name, "start", {"profile": profile, "analysis": analysis, "user_id": context.conversation_id})

            mastery = profile.get("mastery", 0)
            correctness = analysis.get("correctness", "incorrect")

            # Simple logic to update mastery score
            if correctness == "correct":
                mastery += 10
            elif correctness == "partial":
                mastery += 5
            else:
                mastery -= 5

            mastery = max(0, min(100, mastery)) # Clamp mastery between 0 and 100

            # Prepare the updated profile
            updated_profile = profile.copy()
            updated_profile["mastery"] = mastery
            updated_profile["last_errors"] = analysis.get("errorsfound", [])
            updated_profile["last_seen"] = datetime.utcnow().isoformat()
            
            # Save the updated profile back to persistent state
            context.state["profile"] = updated_profile 

            log_event(self.name, "complete", {"updated_profile": updated_profile})
            
            return {
                "updated_profile": updated_profile,
                "message": "Profile updated successfully."
            }
        except Exception as e:
            log_event(self.name, "error", {"error": str(e)})
            raise


# Agent 3: Teaching Strategy Selector (Stateless, no LLM)
class TeachingStrategySelector(Agent):
    """Selects an optimal teaching strategy based on profile and context."""
    def __init__(self):
        super().__init__(name="Agent3_TeachingStrategySelector",
                         description="Selects optimal teaching strategy based on user profile and context.")
    
    # This agent is purely logic-based, so it does not need an LLM.
    def run(self, inputs, context=None):
        try:
            profile = inputs.get("profile", {})
            emotional_state = inputs.get("emotional_state", "neutral")

            log_event(self.name, "start", {"profile": profile, "emotional_state": emotional_state})

            mastery = profile.get("mastery", 0)

            # Logic-based strategy selection
            if mastery < 40:
                strategy = f"Foundational, example-driven teaching (Mastery: {mastery}). Focus on core concepts."
            elif mastery < 80:
                strategy = f"Balanced teaching with guided exercises (Mastery: {mastery}). Reinforce and challenge."
            else:
                strategy = f"Advanced challenges with minimal hints (Mastery: {mastery}). Push for optimization."

            # Adapt strategy based on emotion
            if emotional_state == "frustrated":
                strategy += " Provide extra encouragement and simplify the next step."

            log_event(self.name, "complete", {"strategy": strategy})

            return {
                "teaching_strategy": strategy,
                "recommendation": "Adapt content generation accordingly."
            }
        except Exception as e:
            log_event(self.name, "error", {"error": str(e)})
            raise


# Agent 4: Content Generator Teacher (Stateless, uses LLM)
class ContentGeneratorTeacher(Agent):
    """Generates personalized teaching content using an LLM."""
    def __init__(self, llm: GenerativeModel): # **MODIFIED**: Added llm
        super().__init__(name="Agent4_ContentGeneratorTeacher",
                         description="Generates personalized teaching content: explanations, hints and exercises.")
        self.llm = llm # **MODIFIED**: Store LLM client

    def run(self, inputs, context=None):
        try:
            strategy = inputs.get("teaching_strategy", "")
            profile = inputs.get("profile", {})

            log_event(self.name, "start", {"strategy": strategy, "profile": profile})

            # **MODIFIED**: Replaced mocks with actual LLM call
            prompt = f"""
            You are a helpful and patient DSA Teaching Assistant.
            A student has a current mastery level of {profile.get('mastery', 0)}.
            Your active teaching strategy is: "{strategy}".
            
            Please generate a JSON object with the following keys:
            - "explanation": A concise, pedagogical explanation aligned with the teaching strategy.
            - "hints": A list of 2-3 helpful hints for the student's next attempt or problem.
            - "practice_problems": A list of 2 relevant practice problem titles (e.g., "Two Sum", "Valid Parentheses").
            
            Respond ONLY with the raw JSON object.
            """
            
            # Call the generative model
            response = self.llm.generate_content(prompt)
            output = parse_llm_json(response)

            if "error" in output:
                raise Exception(f"LLM parsing failed: {output.get('raw_text')}")

            log_event(self.name, "complete", output)
            return output
        except Exception as e:
            log_event(self.name, "error", {"error": str(e)})
            raise


# Agent 5: Progress Evaluator Logger (Stateless, no LLM)
class ProgressEvaluatorLogger(Agent):
    """Evaluates progress, logs metrics, and recommends next steps."""
    def __init__(self):
        super().__init__(name="Agent5_ProgressEvaluatorLogger",
                         description="Evaluates progress, logs metrics, and recommends next steps.")

    # This agent is also logic-based.
    def run(self, inputs, context=None):
        try:
            profile = inputs.get("profile", {})
            session_metrics = inputs.get("session_metrics", {})

            log_event(self.name, "start", {"profile": profile, "session_metrics": session_metrics})

            mastery = profile.get("mastery", 0)
            progress_message = f"Current mastery level: {mastery}. Keep up the good work!"

            # Logic-based next steps
            if mastery < 50:
                next_steps = ["Review basics and exercises."]
            else:
                next_steps = ["Attempt advanced problems and projects."]

            output = {
                "progress_message": progress_message,
                "next_steps": next_steps,
            }
            log_event(self.name, "complete", output)
            return output
        except Exception as e:
            log_event(self.name, "error", {"error": str(e)})
            raise


# --- Orchestrator ---

class RootAgentOrchestrator:
    """Orchestrates the 5-agent pipeline."""
    def __init__(self, gemini_model: GenerativeModel): # **MODIFIED**: Requires model
        # **MODIFIED**: Store the model
        self.model = gemini_model
        
        # Setup persistent storage for Agent 2
        self.storage_dir = STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        logging.info(f"Using file system storage at: {self.storage_dir}")
        self.agent2_storage = InFileSystemStorage(storage_path=self.storage_dir)
        
        # **MODIFIED**: Pass the LLM to the agents that need it
        self.agent1_runner = InMemoryRunner(UserInteractionAnalyzer(llm=self.model))
        self.agent4_runner = InMemoryRunner(ContentGeneratorTeacher(llm=self.model))

        # Agents 2, 3, and 5 don't need the LLM
        self.agent2_runner = PersistentRunner(
            agent=LearningProfileManager(),
            storage=self.agent2_storage
        )
        self.agent3_runner = InMemoryRunner(TeachingStrategySelector())
        self.agent5_runner = InMemoryRunner(ProgressEvaluatorLogger())

        self.max_retries = 3

    def _call_with_retries(self, runner, inputs, agent_name, conversation_id=None):
        """Helper function to call an agent runner with retries."""
        for attempt in range(1, self.max_retries + 1):
            try:
                log_msg = f"Calling {agent_name}, attempt {attempt}"
                if conversation_id:
                    log_msg += f" (user_id: {conversation_id})"
                logging.info(log_msg)

                if isinstance(runner, PersistentRunner):
                    # Persistent runners require a conversation_id
                    if not conversation_id:
                        raise ValueError(f"PersistentRunner for {agent_name} requires a conversation_id.")
                    result = runner.run(inputs, conversation_id=conversation_id)
                else:
                    # In-memory runners don't
                    result = runner.run(inputs)
                
                logging.info(f"{agent_name} call succeeded")
                return result
            except Exception as e:
                logging.error(f"{agent_name} call failed on attempt {attempt}; error: {e}")
                if attempt == self.max_retries:
                    logging.error(f"{agent_name} exhausted all retries")
                    raise
                sleep(1) # Wait 1 second before retrying

    def run_pipeline(self, user_input, user_id: str, emotional_state="neutral"):
        """Runs the full 5-agent pipeline."""
        try:
            # 1. Analyze input (Uses LLM)
            analysis = self._call_with_retries(self.agent1_runner, user_input, "UserInteractionAnalyzer")

            # 2. Update profile (Stateful, no LLM)
            profile_input = {"analysis": analysis}
            profile_update = self._call_with_retries(
                self.agent2_runner, 
                profile_input, 
                "LearningProfileManager", 
                conversation_id=user_id
            )
            updated_profile = profile_update.get("updated_profile", {})

            # 3. Select strategy (Logic-based, no LLM)
            strategy_input = {"profile": updated_profile, "emotional_state": emotional_state}
            strategy_info = self._call_with_retries(self.agent3_runner, strategy_input, "TeachingStrategySelector")

            # 4. Generate content (Uses LLM)
            content_input = {"teaching_strategy": strategy_info.get("teaching_strategy", ""), "profile": updated_profile}
            content = self._call_with_retries(self.agent4_runner, content_input, "ContentGeneratorTeacher")

            # 5. Evaluate progress (Logic-based, no LLM)
            progress_input = {"profile": updated_profile, "session_metrics": {"hints_requested": user_input.get("hints_requested", 0)}}
            progress_info = self._call_with_retries(self.agent5_runner, progress_input, "ProgressEvaluatorLogger")

            # Compile the final response
            response = {
                "analysis": analysis,
                "profile_update": updated_profile,
                "teaching_strategy": strategy_info,
                "content": content,
                "progress_update": progress_info,
                "status": "success"
            }
            logging.info(f"Pipeline completed successfully for user_id: {user_id}")
            return response

        except Exception as e:
            logging.error(f"Pipeline failed for user_id: {user_id}: {e}")
            return {
                "status": "failure",
                "error": str(e)
            }


# --- Sample Usage (Demonstrating Persistence & LLM) ---
if __name__ == "__main__":
    
    # **MODIFIED**: Configure genai with your API key
    # In Kaggle, you would use: from kaggle_secrets import UserSecretsClient
    # user_secrets = UserSecretsClient()
    # api_key = user_secrets.get_secret("GEMINI_API_KEY")
    # genai.configure(api_key=api_key)
    
    # !! IMPORTANT !!
    # !! Replace "YOUR_API_KEY_HERE" with your actual Gemini API key
    # !! or use Kaggle secrets as shown above.
    API_KEY = "YOUR_API_KEY_HERE" 
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("="*50)
        print("ERROR: Please update the 'API_KEY' variable in the ")
        print("__main__ block with your Gemini API key to run this demo.")
        print("="*50)
        exit()
        
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        print(f"Error configuring API key: {e}. Please set it in the __main__ block.")
        exit()

    # Setup model
    llm = genai.GenerativeModel('gemini-1.5-pro-latest')

    # **MODIFIED**: Pass the model to the orchestrator
    orchestrator = RootAgentOrchestrator(gemini_model=llm)
    
    sample_user_id = "user_001_llm_test"
    emotional_state = "neutral"

    sample_input = {
        "attemptid": "abc123",
        "code_submission": "def two_sum(nums, target):\n  for i in range(len(nums)):\n    for j in range(i + 1, len(nums)):\n      if nums[i] + nums[j] == target:\n        return [i, j]",
        "language": "python",
        "problem_metadata": {"problem_id": "two_sum", "difficulty": "easy", "test_cases": []},
        "time_taken": 20,
        "hints_requested": 1,
        "explanation": "I used a nested loop to check every pair of numbers.",
        "userconfidence": "medium"
    }

    # --- Run 1 for user_001 ---
    print(f"--- Running pipeline for user: {sample_user_id} (Run 1) ---")
    result1 = orchestrator.run_pipeline(sample_input, sample_user_id, emotional_state)
    print(json.dumps(result1, indent=2))

    # --- Run 2 for user_001 ---
    print(f"\n--- Running pipeline for user: {sample_user_id} (Run 2, same user) ---")
    print("Notice how the mastery score will increase and the strategy adapts...")
    sample_input["attemptid"] = "abc124" # New attempt
    sample_input["userconfidence"] = "high"
    result2 = orchestrator.run_pipeline(sample_input, sample_user_id, emotional_state)
    
    # Print only the relevant parts for the second run to show the change
    print("\n--- Result 2 (Profile Update) ---")
    print(json.dumps(result2.get("profile_update"), indent=2))
    print("\n--- Result 2 (Teaching Strategy) ---")
    print(json.dumps(result2.get("teaching_strategy"), indent=2))