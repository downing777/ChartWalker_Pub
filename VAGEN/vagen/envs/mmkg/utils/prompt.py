def system_prompt():
    """System-level instruction for MMKG environment."""

    base = """You are an intelligent agent navigating a multi-modal, multi-level Knowledge Graph (MMKG).
    Your goal is to explore the graph and find the correct answer for a given query.

    1. Identifying what specific information is required to answer the query.
    2. Navigating to entities that can provide this information.
    3. Extracting concrete facts or numerical values from evidence.
    4. Stopping once all required information has been collected and the query can be answered.

    Exploration actions that do not help fill missing required information are discouraged.

    At each step, you can take one action from Valid action to interact with the KG, 
    carefully reason about which action is most suitable to progress toward solving the query.
    """
    return base

"""
You will receive observations describing:
- The query you are solving
- Candidate start entities(List of entity names)

Once you have selected the start entity from the candidates list
- Current entity and its description
- Available relations (edges) to connected entities 
- Already visited entities
- Visited evidence sources(IDs)
- Memory: Information extracted from visited evidence sources
- Last visited original information(text or image)

- For the `stop` action, you may have collected enough information and output your final answer after the stop operator, e.g. `<answer>stop final_answer_text</answer>`
  Example:
  - `<answer>stop The answer is Paris</answer>`  (output final answer when stopping)
"""


def init_observation_template(observation="",start_candidates = ""):
    """Template for the initial observation at the start of an episode."""
    return f"""[Initial Graph Observation]:
{observation}

Valid action: `start`
Candidate Start Entities: {start_candidates}

Action Gramma
- For action `start`, you must choose one of entity name from the candidate start entities to start this search.If there's only one entity in the list, then start from it:
    Example:
    -"[Observation] Candidate Start Entities ["Country X"]"
    - `<answer>start CountryX</answer>`  (choose CountryX as the starting entity)
"""


def action_template(observation=""):
    """Template for intermediate steps (after first action)."""
    return f"""
After your last response, 

[Current Graph State]:
{observation}
Now, based on the current graph context, reasoning, observation and your exploration history,
decide your next action from: `edge_search`, `move`, `backward`.

WARNING: DO NOT REPEAT YOUR ACTION HISTORY OR CONDUCT ILLEGAL ACTION

Action grammar:
- `edge_search <int>`, you must supply an int argument to indicate the index of relation listed under the **[Searchable Relations]**: 
  Example:
  - `<answer>edge_search 1</answer>`  (inspect relation 1 in the Searchable Relations)
  Warning: search [Forbidden Relations] would get punished
- For action `move`, Move to the relation's target entity that is most likely to provide more required information.
  You can move to any relation's traget entity(both in [Searchable Relations] and [Forbidden Relations — DO NOT SEARCH])
  Example:
  - `<answer>move 1</answer>`  (move to relation 1's target entity)
- For action `backward`, you should supply one of the visited entity name to move back to
  Example:
  - `<answer>backward CountryY</answer>`  (move back to entity named CountryY, which is already visited)
"""

def stop_template(observation=""):
    return f"""
After your last response, you have collected enough information or reach the max stops to answer the question.
Please return your final answer based your observation and the source information you have collected:

{observation}

Now you can only take action `stop`:
- For the `stop` action, you may have collected enough information and output your final answer for the query after the stop operator,
  Example:
  - `<answer>stop final_answer_text</answer>`  (output final answer when stopping)

"""


def format_prompt(max_actions_per_step, prompt_format="free_think"):
    """Generate format prompt based on the specified format"""
    if prompt_format == "free_think":
        return free_think_format_prompt(max_actions_per_step)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")

def free_think_format_prompt(max_actions_per_step):
    """Generate format prompt for free_think format"""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time.
You should first give your reasoning, and then your answer.
Your response should be in the format of:
<think>...</think><answer>...</answer>"""
    
    return base_prompt


