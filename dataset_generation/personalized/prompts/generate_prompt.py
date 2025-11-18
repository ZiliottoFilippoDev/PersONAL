import json
import random
from personalized.generate_graph import gen_ownership
from personalized.utils.graph_utils import convert_ownership_structure
from itertools import chain

# GUIDE
# Easy: 2-5 objects, 5 summaries, one-to-one correspondence between objects and people.
# Medium: 5-7 objects, 5 summaries, at least one person owns two objects.
# Hard: 7-9 objects, 6 summaries, at least one person owns two objects, at least one object owned by two people.


PROMPT_EASY = """You are given a JSON list of objects, each representing an item in a house.  
Each object includes details about the object's category, unique identifier, floor, description, position, and room (if available).

### Input Example:
[
  {
    "object_category": "bed",
    "object_id": "bed_001",
    "room": "bedroom",
    "floor_id": 1,
    "description": ["a blue and white king-sized bed near the window"],
    "position": [2.5, 3.0, 7.2],  // [x, y (height), z]
  },
  {
    "object_category": "refridgerator",
    "object_id": "refridgerator_24",
    "room": "kitchen",
    "floor_id": 1,
    "description": ["a black stainless steel refridgerator"],
    "position": [1.5, 3.0, 2.3],  // [x, y (height), z]
  },
  ...
]

### Task:
1. **Object Selection:**  
   - From the input list, select **between MIN_OBJECTS and MAX_OBJECTS objects**. You have to create a summary that includes details about these objects.  
   - Ensure that across N_SUMMARIES summaries, the object combinations are as distinct as possible.
   - For each summary aim for spatial diversity by picking objects from diverse rooms or zones.

2. **Summary Creation:**  
   - Create N_SUMMARIES different summaries.  
   - For ownership references, replace real names with placeholders like <person1>, <person2>, ..., up to <personM>, where M is the number of chosen objects (e.g., "the comfortable wooden chair owned by <person1>").
   - Make sure that for each summary there is a one-to-one correspondece between <person_i> and <object_i>.
   - Each summary should be a natural, flowing paragraph that includes details (floor, room, object attributes) and ownership information.
  
3. **Output Format:**  
   - The output **must be a raw JSON object**, without any Markdown formatting (e.g., no triple backticks or extra text).  
   - The JSON should have the following structure:
  {
    "summaries": [
      {
      "selected_items": [
        {"object_id": "bed_001", "owner": "<person1>"},
        {"object_id": "refridgerator_24", "owner": "<person2>"}
      ],
      "summary": "In the first-floor bedroom, a blue and white king-sized bed near the window belongs to <person1>, while <person2> owns a black stainless-steel refrigerator in the kitchen.",      
      "extracted_summary": "<person1> owns a blue and white king-sized bed near the window in the bedroom. <person2> owns a black stainless steel refrigerator in the kitchen."
      },
      // Additional OTHER_SUMMARIES summaries with possibly different objects combinations
    ]
  }

4. **Additional Instructions:**  
   - **The output MUST be strictly formatted as JSON.** Do not include any introductory/explanatory text or Markdown formatting.  
   - **Each extracted summary must include ownership statements and room locations using "owns", be separated by periods, and exclude any mention of floor details** (e.g., "first-floor").""" 
   
PROMPT_MEDIUM = """You are given a JSON list of objects, each representing an item in a house.  
Each object includes details about the object's category, unique identifier, floor, description, position, and room (if available).

### Input Example:
[
  {
    "object_category": "bed",
    "object_id": "bed_001",
    "room": "bedroom",
    "floor_id": 1,
    "description": ["a blue and white king-sized bed near the window"],
    "position": [2.5, 3.0, 7.2],  // [x, y (height), z]
  },
  {
    "object_category": "refridgerator",
    "object_id": "refridgerator_24",
    "room": "kitchen",
    "floor_id": 1,
    "description": ["a black stainless steel refridgerator"],
    "position": [1.5, 3.0, 2.3],  // [x, y (height), z]
  },
  ...
]

### Task:
1. **Object Selection:**  
   - From the input list, select **between MIN_OBJECTS and MAX_OBJECTS objects**. You have to create a summary that includes details about these objects.  
   - Ensure that across N_SUMMARIES summaries, the object combinations are as distinct as possible.
   - For each summary aim for spatial diversity by picking objects from diverse rooms or zones.

2. **Summary Creation:**  
   - Create N_SUMMARIES different summaries.  
   - Each summary should be a natural, flowing paragraph that includes details (floor, room, object attributes) and ownership information.  
   - For ownership references, replace real names with placeholders like <person1>, <person2>, ..., up to <personM>, where M is the number of people of the summary.
   - Each summary must include at least one person owning at least two objects.   
   - Ensure that summaries remain coherent and natural, even when people own multiple objects.

3. **Output Format:**  
   - The output **must be a raw JSON object**, without any Markdown formatting (e.g., no triple backticks or extra text).  
   - The JSON should have the following structure:
  {
    "summaries": [
      {
      "selected_items": [
        {"object_id": "bed_001", "owner": "<person1>"},
        {"object_id": "refridgerator_24", "owner": "<person2>"},
        ...
      ],
      "summary": "In the first-floor bedroom, a blue and white king-sized bed near the window belongs to <person1>, while <person2> owns a black stainless-steel refrigerator in the kitchen. ...",      
      "extracted_summary": "<person1> owns a blue and white king-sized bed near the window in the bedroom. <person2> owns a black stainless steel refrigerator in the kitchen. ..."
      },
      // Additional OTHER_SUMMARIES summaries with possibly different objects combinations
    ]
  }

4. **Additional Instructions:**  
   - **The output MUST be strictly formatted as JSON.** Do not include any introductory/explanatory text or Markdown formatting.  
   - If a person owns multiple objects, the extracted summary should still include a separate ownership sentence for each object.
   - **Each extracted summary must include ownership statements and room locations using "owns", be separated by periods, and exclude any mention of floor details** (e.g., "first-floor").""" 
   
PROMPT_HARD = """You are given a JSON list of objects, each representing an item in a house.  
Each object includes details about the object's category, unique identifier, floor, description, position, and room (if available).

### Input Example:
[
  {
    "object_category": "bed",
    "object_id": "bed_001",
    "room": "bedroom",
    "floor_id": 1,
    "description": ["a blue and white king-sized bed near the window"],
    "position": [2.5, 3.0, 7.2],  // [x, y (height), z]
  },
  {
    "object_category": "refridgerator",
    "object_id": "refridgerator_24",
    "room": "kitchen",
    "floor_id": 1,
    "description": ["a black stainless steel refridgerator"],
    "position": [1.5, 3.0, 2.3],  // [x, y (height), z]
  },
  ...
]

### Task:
1. **Object Selection:**  
   - From the input list, select **between MIN_OBJECTS and MAX_OBJECTS objects**. You have to create a summary that includes details about these objects.  
   - Ensure that across N_SUMMARIES summaries, the object combinations are as distinct as possible.
   - For each summary aim for spatial diversity by picking objects from diverse rooms or zones.

2. **Summary Creation:**  
   - Create N_SUMMARIES different summaries.  
   - Each summary should be a natural, flowing paragraph that includes details (floor, room, object attributes) and ownership information.  
   - For ownership references, replace real names with placeholders like <person1>, <person2>, ..., up to <personM>, where M is the number of people of the summary.
   - Each summary must include at least one person owning at least two objects.   
   - Each summary must include at least one object owned by at least two people.
   - When an object is shared between people, this must be clearly integrated as part of the natural summary paragraph (not as a separate sentence at the end).
   - **Across each summary, include at least two different objects that are each jointly owned by two or more people.**
   - Ensure that summaries remain coherent, natural and fluent.

3. **Output Format:**  
   - The output **must be a raw JSON object**, without any Markdown formatting (e.g., no triple backticks or extra text).  
   - The JSON should have the following structure:
  {
    "summaries": [
      {
      "selected_items": [
        {"object_id": "bed_001", "owner": "<person1>"},
        {"object_id": "bed_001", "owner": "<person2>"},
        {"object_id": "refridgerator_24", "owner": "<person2>"},
        {"object_id": "couch_21", "owner": "<person3>"},
        ...
      ],
      "summary": "In the first-floor bedroom, a blue and white king-sized bed near the window, which is shared between <person1> and <person2>, while black stainless-steel refrigerator in the kitchen belongs only to <person2>. ...",      
      "extracted_summary": "<person1> owns a blue and white king-sized bed near the window in the bedroom. <person2> owns a blue and white king-sized bed near the window in the bedroom. <person2> owns a black stainless steel refrigerator in the kitchen. ..."
      },
      // Additional OTHER_SUMMARIES summaries with possibly different objects combinations
    ]
  }

4. **Additional Instructions:**  
   - **The output MUST be strictly formatted as JSON.** Do not include any introductory/explanatory text or Markdown formatting.  
   - If a person owns multiple objects, the extracted summary should still include a separate ownership sentence for each object.
   - If an object is owned by multiple people, the extracted summary should include a separate ownership sentence for each person.
   - **Each extracted summary must include ownership statements and room locations using "owns", be separated by periods, and exclude any mention of floor details** (e.g., "first-floor").""" 
   
   
PROMPT_GRAPH_EASY = """You are a system that transforms a scene description (a list of objects) and a single ownership graph into two outputs:

1. A natural-language summary of the scene.
2. A structured, per-object ownership summary.

**INPUT**
1. Objects  
You are given a JSON array of objects. Each object contains the following keys:  
{
  "object_category": "bed",
  "object_id": "bed_001",
  "room": "bedroom",
  "floor_id": 1,
  "description": ["a blue and white king-sized bed near the window"],
  "position": [2.5, 3.0, 7.2],  // [x, y (height), z]
},
...
{
  "object_category": "refridgerator",
  "object_id": "refridgerator_24",
  "room": "kitchen",
  "floor_id": 1,
  "description": ["a black stainless steel refridgerator"],
  "position": [1.5, 3.0, 2.3],  // [x, y (height), z]
},

2. Ownership  
You are also given a single ownership dictionary. It maps each placeholder person (e.g., <person1>, <person2>, etc.) to a list of object_ids that they own.  
Example:  
{  
  "<person1>": ["bed_001"],  
  "<person2>": ["bed_001", "refrigerator_24"]  
}

**TASK**
Your job is to produce exactly one pair of:

- Ownership Dictionary
- Natural Summary  
- Extracted Summary

A. Ownership Dictionary
- Create a list of selected items, where each item is a dictionary with "object_id" and "owner" keys.
- The "object_id" should match the object_id from the input objects.

B. Natural Summary  
- Write one coherent paragraph (approximately 5 to 10 lines).  
- Maintain clarity and naturalness in the writing.
- Integrate:  
  - The room and floor for each object.  
  - The descriptive content from the "description" field.  
  - The ownership information, using <personX> placeholders (where X starts from 1)
  - Embed the ownership information inline, making a fluent and natural reference to the objects and their owners.
- Use only the information provided. You may add minimal connecting words to ensure the paragraph flows well, but do not hallucinate new facts.

C. Extracted Summary  
- Produce a list of standalone sentences, one for each ownership relation.  
- Each sentence must follow this format:  
  "<personX> owns a [...] in the [room]."  
- Do not mention the floor.  
- Each sentence has to be natural and flowing.
- If two people share one object, list two separate sentences (one per person).  
- If one person owns multiple objects, list each object in a separate sentence.
- Each single object ownership should be contained in one sentence only, without dots in the middle of the sentence.

**OUTPUT FORMAT**
Return exactly one **raw JSON object** with the following structure, without any Markdown formatting (e.g., no triple backticks or extra text).:
{
  "summaries": [
    {
      "selected_items": [
        {"object_id": "bed_01", "owner": "<person1>"},
        {"object_id": "bed_20", "owner": "<person1>"},
        {"object_id": "refridgerator_24", "owner": "<person2>"},
        {"object_id": "couch_21", "owner": "<person3>"},
        ...
      ],
      "summary": "<Your natural paragraph>",
      "extracted_summary": [
        "<person1> owns a blue and white king-sized bed in the bedroom",
        "<person2> owns a stainless steel refridgerator in the kitchen",
        ....
      ]
    }
  ]
  ]
}

Now, based on the input provided above, generate the output JSON containing both the natural summary and the extracted summary, as described. Do not include any extra text."""

PROMPT_GRAPH_MEDIUM = """You are a system that transforms a scene description (a list of objects) and a single ownership graph into two outputs:

1. A natural-language summary of the scene.
2. A structured, per-object ownership summary.

**INPUT**
1. Objects  
You are given a JSON array of objects. Each object contains the following keys:  
{
  "object_category": "bed",
  "object_id": "bed_001",
  "room": "bedroom",
  "floor_id": 1,
  "description": ["a blue and white king-sized bed near the window"],
  "position": [2.5, 3.0, 7.2],  // [x, y (height), z]
},
...
{
  "object_category": "refridgerator",
  "object_id": "refridgerator_24",
  "room": "kitchen",
  "floor_id": 1,
  "description": ["a black stainless steel refridgerator"],
  "position": [1.5, 3.0, 2.3],  // [x, y (height), z]
},

2. Ownership  
You are also given a single ownership dictionary. It maps each placeholder person (e.g., <person1>, <person2>, etc.) to a list of object_ids that they own.  
Example:  
{  
  "<person1>": ["bed_001"],  
  "<person2>": ["bed_001", "refrigerator_24"]  
}

**TASK**
Your job is to produce exactly one pair of:

- Ownership Dictionary
- Natural Summary  
- Extracted Summary

A. Ownership Dictionary
- Create a list of selected items, where each item is a dictionary with "object_id" and "owner" keys.
- The "object_id" should match the object_id from the input objects.

B. Natural Summary  
- Write one coherent paragraph (approximately 5 to 10 lines).  
- Maintain clarity and naturalness in the writing.
- Summary should include:  
  - The room and floor of the current objects.  
  - The descriptive content from the "description" field.  
  - The ownership information, using <personX> placeholders (where X starts from 1).
  - **Aggregate objects**: if a person owns multiple items, try to combine them (e.g. “owns two chairs: one in the ...”).
  - **Aggregate people**: if two or more people own the same object, try to mention them together (e.g. "<person1> and <person2> share a ...").
  - **Aggregate areas**: if multiple objects are in the same area, anchor ownership details togheter (e.g. "In the kitchen, <person1> owns ... .").
  - Mix compound sentences and subordinate clauses to hide the underlying one-to-one correspondence.
  - Embed the ownership information inline, so that the narrative reads like a tour.
- Use only the information provided. You may add minimal connecting words to ensure the paragraph flows well, but DO NOT HALLUCINATE new facts.

C. Extracted Summary  
- Produce a list of standalone sentences, one for each ownership relation.  
- Each sentence must follow this format:  
  "<personX> owns a [...] in the [room]"  
- Each sentence has to be natural and flowing.
- Do not mention the floor.  
- If two people share one object, list two separate sentences (one per person).  
- If one person owns multiple objects, list each object in a separate sentence.
- Each single object ownership should be contained in one sentence only, without dots in the middle of the sentence.

**OUTPUT FORMAT**
Return exactly one **raw JSON object** with the following structure, without any Markdown formatting (e.g., no triple backticks or extra text).:
{
  "summaries": [
    {
      "selected_items": [
        {"object_id": "bed_01", "owner": "<person1>"},
        {"object_id": "bed_20", "owner": "<person1>"},
        {"object_id": "refridgerator_24", "owner": "<person2>"},
        {"object_id": "couch_21", "owner": "<person3>"},
        ...
      ],
      "summary": "<Your natural paragraph>",
      "extracted_summary": [
        "<person1> owns a blue and white king-sized bed in the bedroom",
        "<person2> owns a stainless steel refridgerator in the kitchen",
        ....
      ]
    }
  ]
}

Now, based on the input provided above, generate the output JSON containing both the natural summary and the extracted summary, as described. Do not include any extra text."""


def generate_prompt(object_json, LEVEL="easy", N_SUMMARIES=6, MIN_OBJECTS=2, MAX_OBJECTS=6):
    """
    Generates a prompt for summarizing a list of objects in a house.

    Args:
        object_list (list): A list of dictionaries representing objects in a house.

    Returns:
        str: A formatted prompt string.
    """
    
    # Convert the object list to JSON format
    object_list_json = json.dumps(object_json, indent=2)
    
    if LEVEL in "easy":
        PROMPT = PROMPT_EASY
        
    elif LEVEL in "medium":
        PROMPT = PROMPT_MEDIUM
        
    elif LEVEL in "hard":
        PROMPT = PROMPT_HARD
    
    # Replace placeholders in the prompt with actual values
    prompt = PROMPT
    prompt = prompt.replace("N_SUMMARIES", str(N_SUMMARIES))
    prompt = prompt.replace("MIN_OBJECTS", str(MIN_OBJECTS))
    prompt = prompt.replace("MAX_OBJECTS", str(MAX_OBJECTS))
    prompt = prompt.replace("OTHER_SUMMARIES", str(int(N_SUMMARIES) - 1))
    
    # Combine the prompt and the object list
    return prompt + "\n\n### Input:\n" + object_list_json + "\n\n### Output:\n"


def generate_prompt_from_graph(object_json, LEVEL="easy"):
    """
    Generates a prompt for summarizing a bipartite graph of objects and people.

    Args:
        object_json (list): A list of dictionaries representing objects in a house.
        LEVEL (str): The difficulty level of the prompt. Options are "easy", "medium", "hard".
        N_SUMMARIES (int): The number of summaries to generate.

    Returns:
        str: A formatted prompt string.
    """
    
      # Convert the object list to JSON format
    random.shuffle(object_json)
    
    if LEVEL == "easy":
        mu = 1.0
        overlap = 0.0
        min_number_of_people, max_number_of_people = 2, len(object_json) if len(object_json) <= 5 else 5
        max_objects_per_person = 1
        max_number_of_objects = len(object_json)
    elif LEVEL == "medium":
        mu = random.uniform(1.0, 3.0)
        overlap = random.uniform(0.0, 0.05)
        min_number_of_people, max_number_of_people = 3, 6
        max_objects_per_person = 3
        max_number_of_objects = 7
    elif LEVEL == "hard":
        mu = random.uniform(2.0, 4.0)
        overlap = random.uniform(0.2, 0.4)
        min_number_of_people, max_number_of_people = 4, 8
        max_objects_per_person = 3
        max_number_of_objects = 10 if len(object_json) > 10 else len(object_json)
    
    object_json = object_json[:max_number_of_objects]  # Limit the number of objects to max_number_of_objects
    
    g_ownership, g_metrics = gen_ownership(
      object_json, 
      mu=mu,
      overlap=overlap,
      difficulty=LEVEL,
      min_number_of_people=min_number_of_people,
      max_number_of_people=max_number_of_people,
      max_objects_per_person=max_objects_per_person,
    )
    
    # Take only values of ownership and have a unique list of object_ids
    unique_ids = set(chain.from_iterable(g_ownership.values()))    
    object_json_graph = [o for o in object_json if o["object_id"] in unique_ids]
    object_json_list = json.dumps(object_json_graph, indent=2)
    
    # Selected items
    # selected_items = convert_ownership_structure(g_ownership)
    
    # In the case of empty nodes (medium case)
    g_ownership = compact_person_dict(g_ownership)
    
    # Attach prompt 
    PROMPT_GRAPH = (
        PROMPT_GRAPH_EASY if LEVEL == "easy"
        else PROMPT_GRAPH_MEDIUM if LEVEL == "medium"
        else PROMPT_GRAPH_MEDIUM if LEVEL == "hard"
        else PROMPT_GRAPH_MEDIUM  # fallback
    )    
    prompt = PROMPT_GRAPH + "\n\n**Input**:\n Objects:\n" + object_json_list + "\n\nOwnership:\n" + json.dumps(g_ownership, indent=2) + "\n\n**Output:**\n"
    return {
        "prompt": prompt,
        "ownership": g_ownership,
        "metrics": g_metrics
    }
        
        
def compact_person_dict(person_dict):
    """
    Removes entries with empty lists and rewrites keys as person1, person2, ... without gaps.
    """
    non_empty = [v for v in person_dict.values() if v]
    return {f'<person{i+1}>': v for i, v in enumerate(non_empty)}