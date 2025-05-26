#####################################################################
# 1. 한 토큰 + 기여 토큰(contributions) 형식에 맞춘 새 프롬프트들
#####################################################################

SYSTEM_SINGLE_TOKEN = """Your input consists of short snippets that contain **exactly one focal token**
enclosed in <<double angle brackets>>. On the next line you will see
    Contribution to <<token>> : ("word₁",w₁), ("word₂",w₂), ...
where each tuple lists a contributing token and its weight
(higher = greater contribution to the feature’s activation on the focal token).

Your task:
• Infer the latent linguistic/semantic property that explains why this feature
  activates on the focal token, given the set of contributing tokens.
• Ignore known systematic biases (e.g. sentence-initial tokens, “\\n”) unless
  they are clearly part of the genuine pattern.
• Provide **one concise sentence** describing the pattern; do **not** offer
  multiple guesses or mention the markers themselves.
• End with exactly one line in the form
  [EXPLANATION]: your-single-sentence-explanation
{prompt}
"""

SYSTEM = """You are a meticulous AI researcher investigating language features.
Each example contains **one** focal token marked with << >> and a line:
   Contribution to <<token>> : ("word",weight), ...
Summarize the common latent property causing activation on the focal token,
taking the contributing tokens into account.

Guidelines:
- Be concise: one clear sentence, no lists of alternatives.
- Ignore systematic biases such as high weights on sentence-initial or newline
  tokens unless they genuinely define the pattern.
- Do not mention the marker symbols or quote the snippets.
- Conclude with:
  [EXPLANATION]: your-single-sentence-explanation

{prompt}
"""

SYSTEM_CONTRASTIVE = """You are a meticulous AI researcher investigating language features.
Input contains positive examples (with <<token>> and contribution list) and
optional counter-examples (no highlighted token).

Task & guidelines (same as above):
• Use both positive and counter-examples to pinpoint the latent property.
• Be concise; ignore known bias tokens.
• End with:
  [EXPLANATION]: your-single-sentence-explanation
"""


COT = """
To better find the explanation for the language patterns go through the following stages:

1.Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.

2. Write down general shared latents of the text examples. This could be related to the full sentence or to the words surrounding the marked words.

3. Formulate an hypothesis and write down the final explanation using [EXPLANATION]:.

"""


#####################################################################
# 1. 새 Example 1
#####################################################################
EXAMPLE_1 = """
Example 1:  and he was over the <<moon>> to find

Example 2:  we’ll be laughing till the cows come <<home>>! Pro

Example 3:  thought Scotland was boring, but really there’s more than meets the <<eye>>! I’d
"""

EXAMPLE_1_ACTIVATIONS = """
Example 1:  and he was over the <<moon>> to find
Contribution to <<moon>> : ("over",4), ("the",2), ("moon",9)

Example 2:  we’ll be laughing till the cows come <<home>>! Pro
Contribution to <<home>> : ("till",4), ("cows",4), ("come",3), ("home",9)

Example 3:  thought Scotland was boring, but really there’s more than meets the <<eye>>! I’d
Contribution to <<eye>> : ("than",3), ("meets",5), ("the",2), ("eye",9)
"""

EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "moon", "home", "eye".
CONTRIBUTING TOKENS: ["over":4,"the":2,"moon":9], ["till":4,"cows":4, "come":3,"home":9], ["than":3,"meets":5,"the":2,"eye":9].

Step 1. Each activating token is the key noun in a well-known idiom
        (“over the moon”, “till the cows come home”, “than meets the eye”).
        The contributing tokens are the other words that complete those idioms.

Step 2. Across examples, the feature fires on positive English idioms whose
        core noun carries the main semantic punch.

Step 3. The highest weight is always on the core noun (“moon”, “home”, “eye”),
        with idiom-specific modifiers receiving secondary weights.
"""
EXAMPLE_1_EXPLANATION = """
[EXPLANATION]: The core noun inside upbeat English idioms conveying positive sentiment.
"""

#####################################################################
# 2. 새 Example 2
#####################################################################
EXAMPLE_2 = """
Example 1:  a river is wide but the ocean is wid<<er>>.

Example 2:  every year you get tall<<er>>," she said

Example 3:  the hole became small<<er>> than before

Example 4:  this lake is deep<<er>> than the pond
"""

EXAMPLE_2_ACTIVATIONS = """
Example 1:  a river is wide but the ocean is wid<<er>>.
Contribution to <<er>> : ("is",1), ("wid",4), ("er",9)

Example 2:  every year you get tall<<er>>," she said
Contribution to <<er>> : ("get",1), ("tall",4), ("er",8)

Example 3:  the hole became small<<er>> than before
Contribution to <<er>> : ("became",1), ("small",4), ("er",9)

Example 4:  this lake is deep<<er>> than the pond
Contribution to <<er>> : ("is",1), ("deep",4), ("er",9)
"""


EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "er", "er", "er", "er".
CONTRIBUTING TOKENS: ["is":1,"wid":4,"er":9], ["get":1,"tall":4,"er":8], ["became":1,"small":4,"er":9], ["is":1,"deep":4,"er":9].

Step 1. The activating token is always the suffix “er”, while the contributing
        tokens are the adjective roots it attaches to.

Step 2. The feature targets the comparative suffix “-er” when attached to
        adjectives that express measurable magnitude.

Step 3. The greatest weight goes to the suffix itself, followed by the adjective
        stem that provides the comparison base.
"""
EXAMPLE_2_EXPLANATION = """
[EXPLANATION]: Comparative “-er” suffix on adjectives of size or extent.
"""

#####################################################################
# 3. 새 Example 3
#####################################################################
EXAMPLE_3 = """
Example 1:  something happening inside my <<house>>", he whispered

Example 2:  it was always contained in a <<box>>", according to experts

Example 3:  people were coming into the smoking <<area>> reserved for staff

Example 4:  Patrick: "why are you getting in the <<way>>?" Later,
"""

EXAMPLE_3_ACTIVATIONS = """
Example 1:  something happening inside my <<house>>", he whispered
Contribution to <<house>> : ("inside",3), ("my",1), ("house",9)

Example 2:  it was always contained in a <<box>>", according to experts
Contribution to <<box>> : ("in",2), ("a",1), ("box",9)

Example 3:  people were coming into the smoking <<area>> reserved for staff
Contribution to <<area>> : ("into",2), ("smoking",3), ("area",8)

Example 4:  Patrick: "why are you getting in the <<way>>?" Later,
Contribution to <<way>> : ("in",2), ("the",1), ("way",8)
"""

EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "house", "box", "area", "way".
CONTRIBUTING TOKENS: ["inside":3,"my":1,"house":9], ["in":2,"a":1,"box":9], ["into":2,"smoking":3,"area":8], ["in":2,"the":1,"way":8].

Step 1. Each activating token is a noun naming a container or space where
        something can be located.

Step 2. Contributing tokens highlight spatial inclusion (“in/inside/into”),
        signalling that the noun denotes a place or receptacle.

Step 3. Highest weights belong to the noun itself, with moderate weights on the
        surrounding prepositions that establish the containment relation.
"""
EXAMPLE_3_EXPLANATION = """
[EXPLANATION]: Nouns that denote places or containers which can hold something inside.
"""


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


def system_single_token():
    return [{"role": "system", "content": SYSTEM_SINGLE_TOKEN}]


def system_contrastive():
    return [{"role": "system", "content": SYSTEM_CONTRASTIVE}]
