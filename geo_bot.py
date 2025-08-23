import base64
import json
import re
from io import BytesIO
from typing import Tuple, List, Optional, Dict, Any, Type

from PIL import Image
from langchain_core.messages import HumanMessage, BaseMessage
from hf_chat import HuggingFaceChat
from mapcrunch_controller import MapCrunchController

# The "Golden" Prompt (v7): add more descprtions in context and task
AGENT_PROMPT_TEMPLATE = """
**Mission:** You are an expert geo-location agent. Your goal is to pinpoint our position in as few moves as possible.

**Current Status**
• Remaining Steps: {remaining_steps}  
• Actions You Can Take *this* turn: {available_actions}

────────────────────────────────
**Core Principles**

1.  **Observe → Orient → Act**  
    Start each turn with a structured three-part reasoning block:  
    **(1) Visual Clues —** plainly describe what you see (signs, text language, road lines, vegetation, building styles, vehicles, terrain, weather, etc.).  
    **(2) Potential Regions —** list the most plausible regions/countries those clues suggest.  
    **(3) Most Probable + Plan —** pick the single likeliest region and explain the next action (move/pan or guess).  

2.  **Navigate with Labels:**  
    - `MOVE_FORWARD` follows the green **UP** arrow.  
    - `MOVE_BACKWARD` follows the red **DOWN** arrow.  
    - No arrow ⇒ you cannot move that way.

3.  **Efficient Exploration:**  
    - **Pan Before You Move:** At fresh spots/intersections, use `PAN_LEFT` / `PAN_RIGHT` first.  
    - After ~2 or 3 fruitless moves in repetitive scenery, turn around.

4.  **Be Decisive:** A unique, definitive clue (full address, rare town name, etc.) ⇒ `GUESS` immediately.

5.  **Final-Step Rule:** If **Remaining Steps = 1**, you **MUST** `GUESS` and you should carefully check the image and the surroundings.

────────────────────────────────
**Context & Task:**
Analyze your full journey history and current view, apply the Core Principles, and decide your next action in the required JSON format.

**Action History**
{history_text}

────────────────────────────────
**JSON Output Format:**More actions
Your response MUST be a valid JSON object wrapped in ```json ... ```.
- For exploration: `{{"reasoning": "...", "action_details": {{"action": "ACTION_NAME"}} }}`
- For the final guess: `{{"reasoning": "...", "action_details": {{"action": "GUESS", "lat": <float>, "lon": <float>}} }}`

**Example (valid exploration)**  
```json
{{
  "reasoning": "…",
  "action_details": {{"action": "PAN_LEFT"}}
}}
```

**Example (final guess)**  
```json
{{
  "reasoning": "…",
  "action_details": {{"action": "GUESS", "lat": -33.8651, "lon": 151.2099}}
}}
```
"""

TEST_AGENT_PROMPT_TEMPLATE_NAIVE = """
**Mission:** You are an expert geo-location agent. Your goal is to pinpoint our position based on the surroundings and your observation history.

**Current Status**
• Actions You Can Take *this* turn: {available_actions}

────────────────────────────────
**Core Principles**

1.  **Observe → Orient → Act**  
    Start each turn with a structured three-part reasoning block:  
    **(1) Visual Clues —** plainly describe what you see (signs, text language, road lines, vegetation, building styles, vehicles, terrain, weather, etc.).  
    **(2) Potential Regions —** list the most plausible regions/countries those clues suggest.  
    **(3) Most Probable + Plan —** pick the single likeliest region and explain the next action (move/pan or guess).  

2.  **Navigate with Labels:**  
    - `MOVE_FORWARD` follows the green **UP** arrow.  
    - `MOVE_BACKWARD` follows the red **DOWN** arrow.  
    - No arrow ⇒ you cannot move that way.

3.  **Efficient Exploration:**  
    - **Pan Before You Move:** At fresh spots/intersections, use `PAN_LEFT` / `PAN_RIGHT` first.  
    - After ~2 or 3 fruitless moves in repetitive scenery, turn around.

4.  **Be Decisive:** A unique, definitive clue (full address, rare town name, etc.) ⇒ `GUESS` immediately.

5.  **Always Predict:** On EVERY step, provide your current best estimate of the location, even if you're not ready to make a final guess.

────────────────────────────────
**Context & Task:**
Analyze your full journey history and current view, apply the Core Principles, and decide your next action in the required JSON format.

**Action History**
{history_text}

────────────────────────────────
**JSON Output Format:**
Your response MUST be a valid JSON object wrapped in ```json ... ```.
{{
  "reasoning": "…",
  "current_prediction": {{
    "lat": <float>,
    "lon": <float>,
    "location_description": "Brief description of predicted location"
  }},
  "action_details": {{"action": action chosen from the available actions}}
}}
**Example **  
```json
{{
  "reasoning": "(1) Visual Clues — I see left-side driving, eucalyptus trees, and a yellow speed-warning sign; the road markings are solid white. (2) Potential Regions — Southeastern Australia, Tasmania, or the North Island of New Zealand. (3) Most Probable + Plan — The scene most likely sits in a suburb of Hobart, Tasmania. I will PAN_LEFT to look for additional road signs that confirm this.",
  "current_prediction": {{
    "lat": -42.8806,
    "lon": 147.3250,
    "location_description": "Hobart suburb, Tasmania, Australia"
  }},
  "action_details": {{
    "action": "PAN_LEFT"
  }}
}}
```

"""

TEST_AGENT_PROMPT_TEMPLATE = """
**Mission:** You are an expert geo-location agent. Your goal is to pinpoint our position based on the surroundings and your observation history.

**Current Status**
• Actions You Can Take *this* turn: {available_actions}

────────────────────────────────
**Core Principles & Strategy**

1.  🧠 **Observe → Orient → Act**
    Start each turn with a structured three-part reasoning block:
    **(1) Visual Clues —** Describe the *entire scene*, not just one object. Describe ONLY what you can clearly see. If uncertain, state your uncertainty.
    **(2) Potential Regions —** List plausible regions. Based on the "Think Like a Geographer" principle, if you have a primary candidate, you **must** also list its neighbors with similar biomes as secondary options.
    **(3) Most Probable + Plan —** Pick the single likeliest region, state your **confidence level (Low, Medium, High)**, and explain your plan. If confidence is not High, your plan should be to find a clue that *differentiates* between your top choices.

2.  🌍 **Think Like a Geographer (Crucial for Accuracy)**
    * **Borders are Fuzzy:** Vegetation, soil color, and architecture don't stop at political lines.
    * **Always Consider Neighbors:** When a strong clue points to Country X, you **MUST** consider its direct neighbors with similar climates as plausible secondary options.
    * **Language is a Strong Hint, Not Absolute Proof:** A word in a specific language is a powerful clue but not 100% definitive, especially in border regions.
    * **Action Logic Check:** Before moving towards a target, ensure your action (`FORWARD`, `BACKWARD`) logically matches the target's direction relative to you.

3.  🎯 **Prioritize Information Gain (The "Clue Hunter" Rule)**
    * **Go for the Best Clue:** Your primary goal is to find **definitive clues**. If you spot a potential source of information (a **road sign, text, a unique building**), your immediate objective is to **move towards it**.
    * **Always Re-evaluate Your Target:** While moving towards an objective (e.g., a distant building), you **must** stay observant of the entire scene. If a *new*, potentially more valuable clue appears (e.g., a sign with text becomes visible), you **must** immediately update your objective to investigate the new, better clue. **Text is almost always more valuable than a generic building.**

4.  🧭 **Smart Exploration in Different Environments**
    * **At Intersections & New Locations:** Always perform a quick panoramic scan (`PAN_LEFT`/`PAN_RIGHT`) *before* moving forward.
    * **In Barren Landscapes:** Do a quick 360° pan first. If you see nothing, `MOVE_FORWARD` 3-4 times. If the scenery *still* hasn't changed, `MOVE_BACKWARD` and explore the other direction.

5.   mechanics **Mechanics & Decisiveness**
    * Use `MOVE_FORWARD` and `MOVE_BACKWARD` to navigate.
    * When you find a unique, definitive clue (a national flag, a full address with country) ⇒ `GUESS` immediately.

6.  📍 **Always Predict**
    On EVERY step, provide your current best estimate of the location.

────────────────────────────────
**Context & Task:**
Analyze your full journey history and current view, apply the Core Principles, and decide your next action in the required JSON format.

**Action History**
{history_text}

────────────────────────────────
**JSON Output Format:**
Your response MUST be a valid JSON object wrapped in ```json ... ```.
{{
  "reasoning": "Your detailed, step-by-step thought process, following the three-part structure.",
  "strategic_summary": {{
    "key_clues": [
      {{
        "clue": "A single, significant piece of evidence found so far in the journey.",
        "inferred_regions": ["A list of regions/countries this specific clue points to."]
      }}
    ],
    "plausible_regions": ["Your final, synthesized list of possible regions after considering the intersection and union of all clues."]
  }},
  "current_prediction": {{
    "lat": <float>,
    "lon": <float>,
    "location_description": "Brief description of predicted location"
  }},
  "action_details": {{
    "action": "action chosen from the available actions",
    "objective": "A concise sentence describing the immediate goal of your action. (e.g., 'Read the text on the distant sign', 'Reach the upcoming intersection to get oriented', 'Escape this monotonous road')."
  }}
}}
**Example **
```json
{{
  "reasoning": "(1) Visual Clues — I see left-side driving, eucalyptus trees, and a yellow speed-warning sign; the road markings are solid white. (2) Potential Regions — The primary candidate is Southeastern Australia or Tasmania. New Zealand's North Island is a secondary possibility. (3) Most Probable + Plan — The scene most likely sits in a suburb of Hobart, Tasmania, with Medium confidence. I will PAN_LEFT to look for additional road signs that can confirm this over mainland Australia.",
  "strategic_summary": {{
    "key_clues": [
      {{
        "clue": "Left-side driving",
        "inferred_regions": ["Australia", "New Zealand", "UK", "South Africa", "Japan"]
      }},
      {{
        "clue": "Eucalyptus trees",
        "inferred_regions": ["Australia"]
      }},
      {{
        "clue": "Yellow, diamond-shaped warning sign",
        "inferred_regions": ["Australia", "New Zealand", "North America"]
      }}
    ],
    "plausible_regions": ["Tasmania, Australia", "Southeastern Australia", "North Island, New Zealand"]
  }},
  "current_prediction": {{
    "lat": -42.8806,
    "lon": 147.3250,
    "location_description": "Hobart suburb, Tasmania, Australia"
  }},
  "action_details": {{
    "action": "PAN_LEFT",
    "objective": "Pan left to find a road sign that can confirm the location is Tasmania."
  }}
}}
```
"""

BENCHMARK_PROMPT = """
Analyze the image and determine its geographic coordinates.
1.  Describe visual clues.
2.  Suggest potential regions.
3.  State your most probable location.
4.  Provide coordinates in the last line in this exact format: `Lat: XX.XXXX, Lon: XX.XXXX`
"""


class GeoBot:
    def __init__(
        self,
        model: Type,
        model_name: str,
        use_selenium: bool = True,
        headless: bool = False,
        temperature: float = 0.0,
    ):
        # Initialize model with temperature parameter
        model_kwargs = {
            "temperature": temperature,
        }

        # Handle different model types
        if model == HuggingFaceChat and HuggingFaceChat is not None:
            model_kwargs["model"] = model_name
        else:
            model_kwargs["model"] = model_name

        try:
            self.model = model(**model_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to initialize model {model_name}: {e}")

        self.model_name = model_name
        self.temperature = temperature
        self.use_selenium = use_selenium
        self.controller = MapCrunchController(headless=headless)

    @staticmethod
    def pil_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.thumbnail((1024, 1024))
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _create_message_with_history(
        self, prompt: str, image_b64_list: List[str]
    ) -> List[HumanMessage]:
        """Creates a message for the LLM that includes text and a sequence of images."""
        content = [{"type": "text", "text": prompt}]

        # Add the JSON format instructions right after the main prompt text
        content.append(
            {
                "type": "text",
                "text": '\n**JSON Output Format:**\nYour response MUST be a valid JSON object wrapped in ```json ... ```.\n- For exploration: `{{"reasoning": "...", "action_details": {{"action": "ACTION_NAME"}} }}`\n- For the final guess: `{{"reasoning": "...", "action_details": {{"action": "GUESS", "lat": <float>, "lon": <float>}} }}`',
            }
        )

        for b64_string in image_b64_list:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_string}"},
                }
            )
        return [HumanMessage(content=content)]

    def _create_llm_message(self, prompt: str, image_b64: str) -> List[HumanMessage]:
        """Original method for single-image analysis (benchmark)."""
        return [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ]
            )
        ]

    def _parse_agent_response(self, response: BaseMessage) -> Optional[Dict[str, Any]]:
        """
        Robustly parses JSON from the LLM response, handling markdown code blocks.
        """
        try:
            assert isinstance(response.content, str), "Response content is not a string"
            content = response.content.strip()
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = content
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Invalid JSON from LLM: {e}\nFull response was:\n{response.content}")
            return None

    def init_history(self) -> List[Dict[str, Any]]:
        """Initialize an empty history list for agent steps."""
        return []

    def add_step_to_history(
        self,
        history: List[Dict[str, Any]],
        screenshot_b64: str,
        decision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a step to the history with proper structure.
        Returns the step dictionary that was added.
        """
        step = {
            "screenshot_b64": screenshot_b64,
            "reasoning": decision.get("reasoning", "N/A") if decision else "N/A",
            "action_details": decision.get("action_details", {"action": "N/A"})
            if decision
            else {"action": "N/A"},
        }
        if decision.get("strategic_summary", None) is not None:
            step["strategic_summary"] = decision.get("strategic_summary")
        history.append(step)
        return step

    def generate_history_text(self, history: List[Dict[str, Any]]) -> str:
        """Generate formatted history text for prompt."""
        if not history:
            return "No history yet. This is the first step."

        history_text = ""
        for i, h in enumerate(history):
            history_text += f"--- History Step {i + 1} ---\n"
            history_text += f"Reasoning: {h.get('reasoning', 'N/A')}\n"
            history_text += (
                f"Action: {h.get('action_details', {}).get('action', 'N/A')}\n\n"
            )
        return history_text

    def get_history_images(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extract image base64 strings from history."""
        return [h["screenshot_b64"] for h in history]

    def execute_agent_step(
        self,
        history: List[Dict[str, Any]],
        remaining_steps: int,
        current_screenshot_b64: str,
        available_actions: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a single agent step: generate prompt, get AI decision, return decision.
        This is the core step logic extracted for reuse.
        """
        history_text = self.generate_history_text(history)
        image_b64_for_prompt = self.get_history_images(history) + [
            current_screenshot_b64
        ]

        prompt = AGENT_PROMPT_TEMPLATE.format(
            remaining_steps=remaining_steps,
            history_text=history_text,
            available_actions=available_actions,
        )

        try:
            message = self._create_message_with_history(
                prompt, image_b64_for_prompt[-1:]
            )
            response = self.model.invoke(message)
            decision = self._parse_agent_response(response)
        except Exception as e:
            print(f"Error during model invocation: {e}")
            decision = None

        if not decision:
            print(
                "Response parsing failed or model error. Using default recovery action: PAN_RIGHT."
            )
            decision = {
                "reasoning": "Recovery due to parsing failure or model error.",
                "action_details": {"action": "PAN_RIGHT"},
                "debug_message": f"{response.content.strip()}",
            }

        return decision

    def execute_test_agent_step(
        self,
        history: List[Dict[str, Any]],
        current_screenshot_b64: str,
        available_actions: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a single agent step: generate prompt, get AI decision, return decision.
        This is the core step logic extracted for reuse.
        """
        history_text = self.generate_history_text(history)
        image_b64_for_prompt = self.get_history_images(history) + [
            current_screenshot_b64
        ]

        prompt = TEST_AGENT_PROMPT_TEMPLATE.format(
            history_text=history_text,
            available_actions=available_actions,
        )

        try:
            message = self._create_message_with_history(
                prompt, image_b64_for_prompt[-1:]
            )
            response = self.model.invoke(message)
            decision = self._parse_agent_response(response)
        except Exception as e:
            print(f"Error during model invocation: {e}")
            decision = None

        if not decision:
            print(
                "Response parsing failed or model error. Using default recovery action: PAN_RIGHT."
            )
            decision = {
                "reasoning": "Recovery due to parsing failure or model error.",
                "action_details": {"action": "PAN_RIGHT"},
                "current_prediction": "N/A",
                "debug_message": f"{response.content.strip() if response is not None else 'N/A'}",
            }

        return decision
    
    def execute_action(self, action: str) -> bool:
        """
        Execute the given action using the controller.
        Returns True if action was executed, False if it was GUESS.
        """
        if action == "GUESS":
            return False
        elif action == "MOVE_FORWARD":
            self.controller.move("forward")
        elif action == "MOVE_BACKWARD":
            self.controller.move("backward")
        elif action == "PAN_LEFT":
            self.controller.pan_view("left")
        elif action == "PAN_RIGHT":
            self.controller.pan_view("right")
        return True

    def test_run_agent_loop(self, max_steps: int = 10, step_callback=None) -> Optional[list[Tuple[float, float]]]:
        history = self.init_history()
        predictions = []
        for step in range(max_steps, 0, -1):
            # Setup and screenshot
            self.controller.setup_clean_environment()
            self.controller.label_arrows_on_screen()

            screenshot_bytes = self.controller.take_street_view_screenshot()
            if not screenshot_bytes:
                print("Failed to take screenshot. Ending agent loop.")
                return None

            current_screenshot_b64 = self.pil_to_base64(
                image=Image.open(BytesIO(screenshot_bytes))
            )
            available_actions = self.controller.get_test_available_actions()
            

           
            # Normal step execution
            decision = self.execute_test_agent_step(
                history, current_screenshot_b64, available_actions
            )

            action_details = decision.get("action_details", {})
            action = action_details.get("action")


            # Add step to history AFTER callback (so next iteration has this step in history)
            self.add_step_to_history(history, current_screenshot_b64, decision)

            current_prediction = decision.get("current_prediction")
            if current_prediction and isinstance(current_prediction, dict):
                current_prediction["reasoning"] = decision.get("reasoning", "N/A")
                predictions.append(current_prediction)
            else:
                # Fallback: create a basic prediction structure
                print(f"Invalid current prediction: {current_prediction}")
                fallback_prediction = {
                    "lat": 0.0,
                    "lon": 0.0,
                    "confidence": 0.0,
                    "location_description": "N/A",
                    "reasoning": decision.get("reasoning", "N/A")
                }
                predictions.append(fallback_prediction)
            
            self.execute_action(action)

        return predictions,history
    
    def run_agent_loop(
        self, max_steps: int = 10, step_callback=None
    ) -> Optional[Tuple[float, float]]:
        """
        Enhanced agent loop that calls a callback function after each step for UI updates.

        Args:
            max_steps: Maximum number of steps to take
            step_callback: Function called after each step with step info
                        Signature: callback(step_info: dict) -> None

        Returns:
            Final guess coordinates (lat, lon) or None if no guess made
        """
        history = self.init_history()

        for step in range(max_steps, 0, -1):
            step_num = max_steps - step + 1
            print(f"\n--- Step {step_num}/{max_steps} ---")

            # Setup and screenshot
            self.controller.setup_clean_environment()
            self.controller.label_arrows_on_screen()

            screenshot_bytes = self.controller.take_street_view_screenshot()
            if not screenshot_bytes:
                print("Failed to take screenshot. Ending agent loop.")
                return None

            current_screenshot_b64 = self.pil_to_base64(
                image=Image.open(BytesIO(screenshot_bytes))
            )
            available_actions = self.controller.get_available_actions()
            print(f"Available actions: {available_actions}")

            # Force guess on final step or get AI decision
            if step == 1:  # Final step
                # Force a guess with fallback logic
                decision = {
                    "reasoning": "Maximum steps reached, forcing final guess.",
                    "action_details": {"action": "GUESS", "lat": 0.0, "lon": 0.0},
                }
                # Try to get a real guess from AI
                try:
                    ai_decision = self.execute_agent_step(
                        history, step, current_screenshot_b64, available_actions
                    )
                    if (
                        ai_decision
                        and ai_decision.get("action_details", {}).get("action")
                        == "GUESS"
                    ):
                        decision = ai_decision
                except Exception as e:
                    print(
                        f"\nERROR: An exception occurred during the final GUESS attempt: {e}. Using fallback (0,0).\n"
                    )
            else:
                # Normal step execution
                decision = self.execute_agent_step(
                    history, step, current_screenshot_b64, available_actions
                )

            # Create step_info with current history BEFORE adding current step
            # This shows the history up to (but not including) the current step
            step_info = {
                "step_num": step_num,
                "max_steps": max_steps,
                "remaining_steps": step,
                "screenshot_bytes": screenshot_bytes,
                "screenshot_b64": current_screenshot_b64,
                "available_actions": available_actions,
                "is_final_step": step == 1,
                "reasoning": decision.get("reasoning", "N/A"),
                "action_details": decision.get("action_details", {"action": "N/A"}),
                "history": history.copy(),  # History up to current step (excluding current)
                "debug_message": decision.get("debug_message", "N/A"),
            }

            action_details = decision.get("action_details", {})
            action = action_details.get("action")
            print(f"AI Reasoning: {decision.get('reasoning', 'N/A')}")
            print(f"AI Action: {action}")

            # Call UI callback before executing action
            if step_callback:
                try:
                    step_callback(step_info)
                except Exception as e:
                    print(f"Warning: UI callback failed: {e}")

            # Add step to history AFTER callback (so next iteration has this step in history)
            self.add_step_to_history(history, current_screenshot_b64, decision)

            # Execute action
            if action == "GUESS":
                lat, lon = action_details.get("lat"), action_details.get("lon")
                if lat is not None and lon is not None:
                    return lat, lon
                else:
                    print("Invalid guess coordinates, using fallback")
                    return 0.0, 0.0  # Fallback coordinates
            else:
                self.execute_action(action)

        print("Max steps reached. Agent did not make a final guess.")
        return None

    def analyze_image(self, image: Image.Image) -> Optional[Tuple[float, float]]:
        image_b64 = self.pil_to_base64(image)
        message = self._create_llm_message(BENCHMARK_PROMPT, image_b64)

        try:
            response = self.model.invoke(message)
            print(f"\nLLM Response:\n{response.content}")
        except Exception as e:
            print(f"Error during image analysis: {e}")
            return None

        content = response.content.strip()
        last_line = ""
        for line in reversed(content.split("\n")):
            if "lat" in line.lower() and "lon" in line.lower():
                last_line = line
                break
        if not last_line:
            return None

        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", last_line)
        if len(numbers) < 2:
            return None

        lat, lon = float(numbers[0]), float(numbers[1])
        return lat, lon

    def take_screenshot(self) -> Optional[Image.Image]:
        screenshot_bytes = self.controller.take_street_view_screenshot()
        if screenshot_bytes:
            return Image.open(BytesIO(screenshot_bytes))
        return None

    def close(self):
        if self.controller:
            self.controller.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
