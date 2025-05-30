SYSTEM = """
You are an expert linguistic detective.  
Each input sentence contains exactly **one** target token wrapped in double braces `{{token}}`.  
Every token that **contributes** to this target’s SAE-feature value appears somewhere **before** it in the sentence; those contributor tokens, even if non-contiguous, are grouped together with the target inside a single pair of angle brackets:  
`<< … contributing-token-k … contributing-token-1 {{target}} >>`.  
Your task is to state, in **one concise English sentence**, the semantic or syntactic phenomenon that this SAE feature captures.  
Return **only** one line, formatted exactly as  
[EXPLANATION]: your-single-sentence-explanation 

{prompt}
"""

COT ="""
Think step-by-step **silently** before answering: 
1. For every example, note the target {{token}}, its activation score, and the listed contributor tokens (ignore any “\n” bias tokens).  
2. Identify patterns shared by contributors and/or targets (word class, modality, discourse role, idiom, polarity, etc.).  
3. Observe how the pattern shifts as activation declines to infer the feature’s core meaning and boundaries.  
4. Draft a hypothesis of what the feature detects.  
5. Compress that hypothesis into ONE well-formed English sentence.  
6. Output **only** that sentence prefixed with “[EXPLANATION]: ”—nothing else."""

EXAMPLE_1 ="""
He was <<over the {{moon}}>> when he got the job.  
She felt <<on top of the {{world}}>> after the final.  
Fans are <<through the {{roof}}>> with excitement today."""

EXAMPLE_1_ACTIVATIONS ="""
Example_1 : He was <<over the {{moon}}>> when he got the job.  
Contribution to token {{moon}} whose feature activation is 2.112 : [("over",4), ("the",2), ("moon",9)]  
Example_2 : She felt <<on top of the {{world}}>> after the final.  
Contribution to token {{world}} whose feature activation is 1.937 : [("top",3), ("of",2), ("the",2), ("world",8)]  
Example_3 : Fans are <<through the {{roof}}>> with excitement today.
Contribution to token {{roof}} whose feature activation is 1.621 : [("through",4), ("the",2), ("roof",7)]"""

EXAMPLE_1_COT_ACTIVATION_RESPONSE ="""
• Highest activations sit on nouns anchoring exuberant idioms (“over the moon”, etc.).  
• Contributors are the idiom’s function words.  
• Activation fades on less-common idioms. → Feature fires on nouns in highly positive idioms."""

EXAMPLE_1_EXPLANATION ="""
[EXPLANATION]: The noun at the heart of upbeat English idioms signalling extreme happiness or excitement."""

EXAMPLE_2 = """
<<Experienced athletes {{can}}>> complete the course in under two hours.  
<<Careful analysis {{may}}>> help resolve the issue.  
<<With a little luck {{might}}>> we avoid delays.
"""

EXAMPLE_2_ACTIVATIONS ="""
Example_1 : <<Experienced athletes {{can}}>> complete the course in under two hours.  
Contribution to token {{can}} whose feature activation is 2.145 : [("Experienced",3), ("athletes",2), ("can",9)]  
Example_2 : <<Careful analysis {{may}}>> help resolve the issue.  
Contribution to token {{may}} whose feature activation is 1.944 : [("Careful",2), ("analysis",2), ("may",9)]  
Example_3 : <<With a little luck {{might}}>> we avoid delays.
Contribution to token {{might}} whose feature activation is 1.512 : [("With",1), ("a",1), ("little",1), ("luck",2), ("might",8)]"""

EXAMPLE_2_COT_ACTIVATION_RESPONSE ="""
• Highest scores attach to modal auxiliaries “can / may / might”.  
• All contributors precede the modal, even if separated.  
• Strength drops from strong ability (“can”) to weak possibility (“might”).  
→ Feature encodes graded modality."""

EXAMPLE_2_EXPLANATION ="""
[EXPLANATION]: Core English modal auxiliaries expressing ability or decreasing degrees of possibility."""

EXAMPLE_3 ="""
The plan sounded great; <<{{however}}>> the budget was gone.  
Sales rose in Q1, <<{{but}}>> Q2 was disappointing.  
It’s a sturdy phone, <<although the weight {{although}}>> is a drawback."""

EXAMPLE_3_ACTIVATIONS = """
Example_1 : The plan sounded great; <<{{however}}>> the budget was gone.  
Contribution to token {{however}} whose feature activation is 2.031 : [("however",8), (";",2)]  
Example_2 : Sales rose in Q1, <<{{but}}>> Q2 was disappointing.  
Contribution to token {{but}} whose feature activation is 1.774 : [("but",8)]  

Example_3 : It’s a sturdy phone, <<although the weight {{although}}>> is a drawback.
Contribution to token {{although}} whose feature activation is 1.423 : [("although",7), (",",2)]"""



EXAMPLE_3_COT_ACTIVATION_RESPONSE ="""
• Tokens are adversative conjunctions.  
• Activation strongest clause-initial (“however”), weaker when embedded (“although”).  
→ Feature marks contrastive connectors."""

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
