parse_prompt = """
    You will be given the list of objects in an apartment, as well as a sequence of actions performed by the agent. Your task is to infer the agent's perception before each action. The agent starts with no knowledge of the environment, so all perceptions must be derived from their visual experience and direct interaction with objects during the sequence.
    When inferring perception, focus only on the objects the agent can see at that moment, and avoid adding unnecessary details. The agent can see objects that are visibly placed, such as those on tables, sofas, or the floor. However, the agent cannot see the contents of containers — such as closets, drawers, stoves, or fridges — until they explicitly open them.
    Some items in the action sequence may already describe the agent's perception (e.g., "see", "notice"), but these are not actions and should be treated purely as indicators of the agent's perception. Additionally, some compound actions can be merged. For instance, a sequence like “close the drawer, proceed to the cabinet and then open it” can be deemed as a single action, as the intermediate perception adds no new information. On the other hand, actions such as "open a container and then close" must be seperated — even if the container is empty — because opening reveals a new perception, such as “the container is empty.”
    Only infer new perception at each step. If the agent's perception does not change between two actions, or if the same perceptual information would be repeated, you may merge the action with its successor. For example, “close the cabinet and move to the bedroom” involves no new perception after closing the cabinet — so the perception should instead be associated with the action that follows, such as entering the bedroom.
    After you have constructed the trajectory (each step consisting of a perception and an action), extract the knowledge from the apartment object list. This knowledge should include only the rooms the agent actually visited — remove any unvisited rooms or their contents from the final output.
    "on the verge of opening the fridge" is also an action, but never go beyond the given action sequence when producing the trajectory.
    You must output the result in JSON format, strictly as:
    {"Trajectory": [{"Perception": A description of the perception before the action, "Action": Each decomposed action in the narrative}, ...], "Knowledge": Part of what's in the apartment without the unvisited rooms}.
"""

hypothesis_prompt = """
    You will be given a description of the apartment, the agent's perception, the agent's current action, the full interaction history, and a list of hypotheses. The hypotheses represent possible beliefs and intentions of the agent — they are candidate explanations for why the agent performed a certain action. For example, a valid hypothesis could be: “Nancy believes the object she wants is in the microwave.”
    You are expected to update each hypothesis based on all the given information. If the list of hypotheses is empty, you must carefully construct and initialize a list of {N} plausible hypotheses.
    When updating each hypothesis, consider the context step by step. For instance, if the agent opens the microwave and finds no salmon there, you should revise the belief accordingly. Likewise, if the agent perceives that the salmon is in the microwave but then closes it, this may indicate that the salmon is not what the agent wants, and the hypothesis should be reframed.
    Each hypothesis should be associated with a likelihood score that reflects how likely the observed action is given that belief. The likelihood should be selected on a discrete scale:
    0 = “Definitely not”
    1 = “Very unlikely”
    … up to
    6 = “Very likely”
    Crucially, your reasoning must be grounded only in the agent's perception. The agent begins with no knowledge of the environment. They do not know what's inside any container until they open it. For example, if the current action is "open the fridge", and the knowledge says the fridge contains a plate, you should not infer that the agent is seeking the plate — because they have not yet perceived its contents. Similarly, if the knowledge says the microwave contains the desired item, you should not preemptively infer that the agent will check the microwave before perceiving it.
    The agent's beliefs are shaped solely by their perception and their internal motivations. You should reason in terms of “what they want”, rather than in terms of specific, unperceived objects. In particular, any object the agent has already encountered and bypassed is definitely not desired. For example, if Nancy closes a closet that contains a plate, you can infer that "Nancy does not want the plate." Likewise, if the agent does not check a container, this likely means they believe the object they're looking for is not in that container. However, failing to check a container does not necessarily imply the agent is not seeking the object it contains—it only indicates that, at that point, the agent did not believe the object was there.
    Do not carry over or track state — your output will be appended to the evolving interaction history. Just focus on producing a concise and up-to-date list of hypotheses that reflect the agent's most recent beliefs.
    Lastly, note that in this hypothetical world, objects and containers are not logically constrained. For example, a remote control may be stored in a microwave or a refrigerator. Always remember: the agent's belief may contradict the objective knowledge of the environment, only ground your hypotheses in the agent's belief and perception.
    You must return the result in the following JSON format: {{"Reason": First think step by step about the situation, "Hypotheses": [{{"Hypothesis": Updated or Initial hypothesis, "Likelihood": The likelihood of the action given the hypothesis}}, ...]}}.
"""

summary_prompt = """
    Based on the weighted hypotheses and the full history, you need to summarize the agent's belief.
    The summary should be concise and to the point, stick to the weighted hypotheses.
    You can try to merge some hypotheses if they are consistent, otherwise try prioritizing the most likely ones.
    Ouput the summary directly.
"""

rejuvenate_prompt = """
    You will be given a list of hypotheses, and the knowledge about the apartment.
    If they lack diversity, you need to paraphrase some hypotheses to make them more diverse.
    You can try change some of the objects in the hypotheses based on the knowledge.
    However, at least one hypothesis of each similar groups should be kept the same.
    Do not change the number of hypotheses.
    You must only give in JSON format a list of Hypotheses.
"""

CoT_prompt = """
    You will be given the knowledge about the apartment, the full history, the agent's perception and the action.
    You need to reason about the agent's belief and intention.
    Crucially, your reasoning must be grounded only in the agent's perception. The agent begins with no knowledge of the environment. They do not know what's inside any container until they open it. For example, if the current action is "open the fridge", and the knowledge says the fridge contains a plate, you should not infer that the agent is seeking the plate — because they have not yet perceived its contents. Similarly, if the knowledge says the microwave contains the desired item, you should not preemptively infer that the agent will check the microwave before perceiving it.    
    When reasoning about the agent's intentions, you should never use external knowledge or make assumptions based on facts unknown to the agent. For example, even if the knowledge indicates that the microwave contains the object being searched for, you must not infer that the agent is likely to check the microwave unless their actions or perception suggest so.
    If the agent opens a container, it implies they believe the object they want might be in it. Conversely, if they did not check a certain container in the room, it often means they believe the object they want is not in that container. However, failing to check a container does not necessarily imply the agent is not seeking the object it contains—it only indicates that, at that point, the agent did not believe the object was there.
    The reasoning should be based on what the agent wants, rather than on specific concrete objects. This distinction is important because the agent may not know where the object is or what exactly it looks like. Therefore, frame your inference in terms of “what they want.” For example, if the agent opens the fridge, they believe what they want is in the fridge, even if the fridge actually contains something unrelated.
    Similarly, if the current action is “open the fridge,” and the knowledge says the fridge contains a plate, this does not mean the agent is seeking the plate—the agent has not yet perceived what's in the fridge. You must not infer intentions based on knowledge the agent has not accessed.
    Objects that the agent has already passed or dismissed are certainly not desired. For instance, if Nancy closes the closet that contains a plate, it can be inferred that Nancy does not want the plate.
    There is no need to keep track of something because your output will be concatenated to the history. Keep the hypotheses concise and reflect the newest belief.
    In this hypothetical world, the object and the place are not necessarily related. For example, the remote control can be in the microwave or refrigerator.
    You can first think step by step, then give the final answer.
    You must only give in JSON format, strictly as {"Reason": Your reasoning, "Thought": The deducted agent's beliefs and intention}.
"""

conclusion_prompt = """
    You will be given the knowledge about the apartment, the full history including the summarized hypotheses of the agent's mind and the query.
    They are about an agent finding an object in the apartment without any prior knowledge.
    You must conform to the query's assumption throughout your reasoning.
    The agent begins with no knowledge of the environment. They do not know what is inside any container until they open it. Therefore, all reasoning about the agent's beliefs must be grounded entirely in their perception and the thoughts that can be deduced from it. Their beliefs may contradict the actual knowledge of the environment, and this is expected, since the agent acts based on what they perceive—not on the truth.
    When reasoning about the agent's intentions, you should never use external knowledge or make assumptions based on facts unknown to the agent. For example, even if the knowledge indicates that the microwave contains the object being searched for, you must not infer that the agent is likely to check the microwave unless their actions or perception suggest so.
    If the agent opens a container, it implies they believe the object they want might be in it. Conversely, if they did not check a certain container in the room, it often means they believe the object they want is not in that container. However, failing to check a container does not necessarily imply the agent is not seeking the object it contains—it only indicates that, at that point, the agent did not believe the object was there.
    The reasoning should be based on what the agent wants, rather than on specific concrete objects. This distinction is important because the agent may not know where the object is or what exactly it looks like. Therefore, frame your inference in terms of “what they want.” For example, if the agent opens the fridge, they believe what they want is in the fridge, even if the fridge actually contains something unrelated, like a plate.
    Similarly, if the current action is “open the fridge,” and the knowledge says the fridge contains a plate, this does not mean the agent is seeking the plate—the agent has not yet perceived what's in the fridge. You must not infer intentions based on knowledge the agent has not accessed.
    Objects that the agent has already passed or dismissed are certainly not desired. For instance, if Nancy closes the closet that contains a plate, it can be inferred that Nancy does not want the plate.
    Lastly, note that in this hypothetical world, objects and containers are not logically constrained. For example, a remote control may be stored in a microwave or a refrigerator. Going to the kitchen does not necessarily mean the agent wants to get some food. Do not use typical knowledge regarding the objects' location to make assumptions about the agent's belief, focus on the agent's perception and the thoughts.
    Always remember: the agent's belief may contradict the objective knowledge of the environment, only ground your hypotheses in the agent's belief and perception.
    You can first think step by step, then give the final answer.
    You must only give in JSON format, strictly as {"Reason": Your reasoning, "Answer": The final answer (a or b)}.
"""

test_input = """
    What's inside the apartment: 
    The apartment consists of a bedroom, a bathroom, a living room, and a kitchen.
    In the bedroom, there is a desk and a coffee table, which holds two wine glasses, two remote controls, a book, a water glass, and a bottle of wine.
    The bathroom features a cabinet, which is currently empty.
    The living room is furnished with a cabinet, a coffee table, a desk, and a sofa. Inside the cabinet, you'll find a bag of chips, a cupcake, a condiment bottle, two plates, three water glasses, and a bottle of wine.
    The kitchen is equipped with a stove, four cabinets, a microwave, a table, and a fridge. Inside the stove, there is a plate. The third cabinet from the left is empty, while the fourth one contains a plate. The microwave houses a salmon. The second cabinet from the left holds a water glass, and the first one is empty. The fridge is stocked with a plate, a cupcake, an apple, and a salmon.
    Actions taken by Nancy: Nancy is situated in the kitchen. She strides towards the microwave, opens it, then closes it. She then proceeds to the third kitchen cabinet, opens and closes it. Following this, she opens and closes the stove. She then moves to the second kitchen cabinet, opens and closes it. Subsequently, she walks towards the bathroom, returns to the kitchen, and finally approaches the fourth kitchen cabinet.
    Question: Which one of the following statements is more likely to be true? (a) Nancy has been trying to get a book. (b) Nancy has been trying to get a cupcake. Please respond with either a or b.
"""

test_trajectory = [{'Perception': 'Nancy is in the kitchen. She sees a stove, four cabinets, a microwave, a table, and a fridge.', 'Action': 'Nancy strides towards the microwave and opens it.'}, {'Perception': 'Inside the microwave, Nancy sees a salmon.', 'Action': 'Nancy closes the microwave and proceeds to the third kitchen cabinet, opens it.'}, {'Perception': 'The third kitchen cabinet is empty.', 'Action': 'Nancy closes the third kitchen cabinet and opens the stove.'}, {'Perception': 'Inside the stove, Nancy sees a plate.', 'Action': 'Nancy closes the stove and opens the second kitchen cabinet.'}, {'Perception': 'Inside the second kitchen cabinet, Nancy sees a water glass.', 'Action': 'Nancy closes the second kitchen cabinet and walks towards the bathroom.'}, {'Perception': 'Nancy is in the bathroom. She sees a cabinet.', 'Action': 'Nancy returns to the kitchen and approaches the fourth kitchen cabinet.'}]