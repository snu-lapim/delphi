SYSTEM = """
You are an expert linguistic detective.  
Each input sentence contains exactly **one** target token wrapped in double braces `{{token}}`.  
Every token that **contributes** to this target’s SAE-feature value appears somewhere **before** it in the sentence; those contributor tokens, even if non-contiguous, are grouped together with the target inside a single pair of angle brackets:  
`<< … contributing-token-k … contributing-token-1 {{target}} >>`.  
Your task is to state, in **one concise English sentence**, the semantic or syntactic phenomenon that this SAE feature captures.

{prompt}
"""

COT ="""
To better find the explanation for the language patterns go through the following stages:
1. For every example, **YOU MUST WRITE DOWN FIRST** list *no more than ten* activation rows that best illustrate the pattern shift (include some high, mid-range, and low activations).  
   For each chosen row write the target {{token}}, its activation score, and the contributor tokens with their contribution scores (ignore any “\\n” bias tokens). 
   **Make sure there are no more than ten rows in total**, and that they are ordered by activation score from highest to lowest.
2. Identify patterns shared by contributors and/or targets (word class, modality, discourse role, idiom, polarity, etc.).  
3. Observe how the pattern shifts as activation declines to infer the feature’s core meaning and boundaries.  
4. Draft a hypothesis of what the feature detects.  
5. Compress that hypothesis into ONE well-formed English sentence.  
6. Write down the final explanation using [EXPLANATION]:."""

EXAMPLE_1 ="""
He was <<over the {{moon}}>> when he got the job.  
She felt <<on top of the {{world}}>> after the final.  
Fans are <<through the {{roof}}>> with excitement today."""

EXAMPLE_1_ACTIVATIONS ="""
Example_1 : He was <<over the {{moon}}>> when he got the job.  
Contribution to token {{moon}} whose feature activation is 2.112 : [("over" : 4), ("the" : 2), ("moon" : 9)]  
Example_2 : She felt <<on top of the {{world}}>> after the final.  
Contribution to token {{world}} whose feature activation is 1.937 : [("top" : 3), ("of" : 2), ("the" : 2), ("world" : 8)]  
Example_3 : Fans are <<through the {{roof}}>> with excitement today.
Contribution to token {{roof}} whose feature activation is 1.621 : [("through" : 4), ("the" : 2), ("roof" : 7)]"""

EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
1. Targets & contributors:
   •[1] {{moon}} (2.112) ← "over" : 4, "the" : 2, "moon" : 9
   •[2] {{world}} (1.937) ← "top" : 3, "of" : 2, "the" : 2, "world" : 8
   •[3] {{roof}} (1.621) ← "through" : 4, "the" : 2, "roof" : 7
2. Every target is a noun inside an exuberant fixed idiom (‘over the moon’, ‘on top of the world’, ‘through the roof’).  
3. Activation drops as the idiom becomes less frequent or vivid (moon > world > roof).  
4. Hypothesis: the feature fires on the core noun of highly positive idioms denoting extreme happiness or excitement.  
5. Compress → 
"""

EXAMPLE_1_EXPLANATION ="""
[EXPLANATION]: The noun at the heart of upbeat English idioms signalling extreme happiness or excitement."""

EXAMPLE_2 = """
<<Experienced athletes {{can}}>> complete the course in under two hours.  
<<Careful analysis {{may}}>> help resolve the issue.  
<<With a little luck {{might}}>> we avoid delays.
"""

EXAMPLE_2_ACTIVATIONS ="""
Example_1 : <<Experienced athletes {{can}}>> complete the course in under two hours.  
Contribution to token {{can}} whose feature activation is 2.145 : [("Experienced" : 3), ("athletes" : 2), ("can" : 9)]  
Example_2 : <<Careful analysis {{may}}>> help resolve the issue.  
Contribution to token {{may}} whose feature activation is 1.944 : [("Careful" : 2), ("analysis" : 2), ("may" : 9)]  
Example_3 : <<With a little luck {{might}}>> we avoid delays.
Contribution to token {{might}} whose feature activation is 1.512 : [("With" : 1), ("a" : 1), ("little" : 1), ("luck" : 2), ("might" : 8)]"""

EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
1. Targets & contributors:
   •[1] {{can}} (2.145) ← "Experienced" : 3, "athletes" : 2, "can" : 9
   •[2] {{may}} (1.944) ← "Careful" : 2, "analysis" : 2, "may" : 9
   •[3] {{might}} (1.512) ← "With" : 1, "a" : 1, "little" : 1, "luck" : 2, "might" : 8
2. All targets are core modal auxiliaries; contributors are the preceding noun or prepositional phrases plus the modal itself.  
3. As modality weakens from strong ability (‘can’) to tentative possibility (‘might’), activation steadily falls (can > may > might).  
4. Hypothesis: the feature detects English modals and grades them by perceived modal strength.  
5. Compress →  
"""
EXAMPLE_2_EXPLANATION ="""
[EXPLANATION]: Core English modal auxiliaries expressing ability or decreasing degrees of possibility."""

EXAMPLE_3 ="""
The plan sounded great; <<{{however}}>> the budget was gone.  
Sales rose in Q1, <<{{but}}>> Q2 was disappointing.  
It’s a sturdy phone, <<although the weight {{although}}>> is a drawback."""

EXAMPLE_3_ACTIVATIONS = """
Example_1 : The plan sounded great<<; {{however}}>> the budget was gone.  
Contribution to token {{however}} whose feature activation is 2.031 : [("however" : 8), (";" : 2)]  
Example_2 : Sales rose in Q1, <<{{but}}>> Q2 was disappointing.  
Contribution to token {{but}} whose feature activation is 1.774 : [("but" : 8)]  
Example_3 : It’s a sturdy phone<<, {{although}}>> the weight is a drawback.
Contribution to token {{although}} whose feature activation is 1.423 : [("," : 2), ("although" : 7)]"""




EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
1. Targets & contributors:
   •[1] {{however}} (2.031) ← “;” : 2, "however" : 8
   •[2] {{but}} (1.774) ← "but" : 8
   •[3] {{although}} (1.423) ← “,” : 2, "although" : 7
2. The targets are adversative conjunctions or discourse markers; contributors are nearby punctuation and the token itself.
3. Clause-initial markers (‘however’) show the highest activation, while embedded concessives (‘although’) show the lowest (however > but > although).
4. Hypothesis: the feature highlights connectors that introduce contrast, with extra weight when they open a new clause.
5. Compress →
"""

EXAMPLE_3_EXPLANATION = """
[EXPLANATION]: Adversative conjunctions and discourse markers that signal contrast or concession, strongest when clause-initial."""



# ---------------------------------------------------------------
# Put your 40 real examples below, in the same format:
# ★★PLACE_40_EXAMPLES_HERE★★
# ---------------------------------------------------------------


def get(item):
    return globals()[item]


def _prompt(n, activations=False, **kwargs):
    starter = (
        get(f"EXAMPLE_{n}") if not activations else get(f"EXAMPLE_{n}_ACTIVATIONS")
    )

    prompt_atoms = [starter]

    return "".join(prompt_atoms)


def _response(n, cot=False, **kwargs):
    response_atoms = []
    if cot:
        response_atoms.append(get(f"EXAMPLE_{n}_COT_ACTIVATION_RESPONSE"))

    response_atoms.append(get(f"EXAMPLE_{n}_EXPLANATION"))

    return "".join(response_atoms)


def example(n, **kwargs):
    prompt = _prompt(n, **kwargs)
    response = _response(n, **kwargs)

    return prompt, response


def system(cot=False):
    prompt = ""

    if cot:
        prompt += COT

    return [
        {
            "role": "system",
            "content": SYSTEM.format(prompt=prompt),
        }
    ]


# def system_single_token():
#     return [{"role": "system", "content": SYSTEM_SINGLE_TOKEN}]


# def system_contrastive():
#     return [{"role": "system", "content": SYSTEM_CONTRASTIVE}]
