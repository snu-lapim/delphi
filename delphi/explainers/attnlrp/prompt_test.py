############################
#  UPDATED PROMPT TEMPLATE #
############################

SYSTEM = """
You are an expert linguistic detective.  
Each input sentence contains exactly **one** target token wrapped in double braces `{{token}}`.  
Every token that **contributes** to this target’s SAE-feature value appears somewhere **before** it in the sentence; those contributor tokens, even if non-contiguous, are grouped together with the target inside a single pair of angle brackets:
`<< … contributing-token-k … contributing-token-1 {{target}} >>`.

Your task is to state, in **one concise English sentence**, the semantic or syntactic phenomenon that this SAE feature captures, **also indicating how the feature’s activation strength varies across clear tiers (e.g., strong > medium > weak)**.

{prompt}
"""

COT = """
To uncover the feature, move through these stages **in order**:

1. **Activation sampling (≤ 10 rows)**  
   — Select up to ten representative rows (high / mid / low).  
   — For each, list the target `{{token}}`, its activation score, and its contributors with their individual scores (omit any “\\n” bias tokens).  
   — Sort rows from highest to lowest activation.

2. **Shared patterns**  
   — Note commonalities among contributors and/or targets (POS, meaning field, discourse role, idiom, polarity, etc.).

3. **Activation-gradient analysis**  
   — Explain how the shared pattern tightens or loosens as scores fall (e.g., core cases at 2-3, peripheral echoes ≈1, noise < 0.5).

4. **Hypothesis drafting**  
   — Propose in prose what linguistic feature the model is detecting and how it is graded.

5. **Compression**  
   — Boil the hypothesis into **ONE well-formed English sentence that already embeds the gradient (strong > medium > weak)**.

6. **Output**  
   — Present only that sentence, prefixed with `[EXPLANATION]:`.
"""

####################################
#  EXAMPLE 1 —  POSITIVE   IDIOMS  #
####################################

EXAMPLE_1 = """
He was <<over the {{moon}}>> when he got the job.  
She felt <<on top of the {{world}}>> after the final.  
Fans are <<through the {{roof}}>> with excitement today.
"""

EXAMPLE_1_ACTIVATIONS = """
Example 1 : He was <<over the {{moon}}>> …      activation 2.112 : [("over":4), ("the":2), ("moon":9)]
Example 2 : She felt <<on top of the {{world}}>> activation 1.937 : [("top":3), ("of":2), ("the":2), ("world":8)]
Example 3 : Fans are <<through the {{roof}}>> … activation 1.621 : [("through":4), ("the":2), ("roof":7)]
"""

EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
1. Activation rows (hi→lo)  
   • {{moon}}  2.112 ← over 4, the 2, moon 9  
   • {{world}} 1.937 ← top 3, of 2, the 2, world 8  
   • {{roof}}  1.621 ← through 4, the 2, roof 7  

2. Shared pattern → each target is the core noun of a fixed “extreme-happiness” idiom.  
   Contributors are obligatory idiom prepositions (“over”, “on top of”, “through”) plus determiners.

3. Gradient → familiarity / vividness:  
     ‘over the moon’ (canonical, very common) ▶ ‘on top of the world’ (still common) ▶ ‘through the roof’ (somewhat looser, can be anger too).  
     Activation mirrors this drop-off.

4. Hypothesis → the feature lights up the nucleus of exuberant positive idioms, with score ≈ idiom prototypicality.

5. Compress →
"""

EXAMPLE_1_EXPLANATION = """
[EXPLANATION]: The feature fires strongest on the noun at the heart of very common euphoric idioms (“over the moon”), slightly less on moderately common ones (“on top of the world”), and weakest on looser or polysemous variants (“through the roof”).
"""

####################################
#  EXAMPLE 2 —  MODAL  STRENGTH    #
####################################

EXAMPLE_2 = """
<<Experienced athletes {{can}}>> complete the course in under two hours.  
<<Careful analysis {{may}}>> help resolve the issue.  
<<With a little luck {{might}}>> we avoid delays.
"""

EXAMPLE_2_ACTIVATIONS = """
… {{can}}   activation 2.145 : [("Experienced":3), ("athletes":2), ("can":9)]  
… {{may}}   activation 1.944 : [("Careful":2), ("analysis":2), ("may":9)]  
… {{might}} activation 1.512 : [("With":1), ("a":1), ("little":1), ("luck":2), ("might":8)]
"""

EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
1. Rows  
   • can 2.145 ← context NP + can 9  
   • may 1.944 ← context NP + may 9  
   • might 1.512 ← PP of luck + might 8  

2. Pattern → target = core modal auxiliary; contributors = preverbal context + the modal.

3. Gradient → perceived modality strength: ability/strong possibility (can) > neutral permission/possibility (may) > tentative possibility (might).

4. Hypothesis → detector grades English modals by pragmatic strength.

5. Compress →
"""

EXAMPLE_2_EXPLANATION = """
[EXPLANATION]: Core English modals are ranked by force, scoring highest for strong ability/possibility (“can”), mid for neutral permission (“may”), and lowest for tentative possibility (“might”).
"""

####################################
#  EXAMPLE 3 —  ADVERSATIVE LINKS  #
####################################

EXAMPLE_3 = """
The plan sounded great; <<{{however}}>> the budget was gone.  
Sales rose in Q1, <<{{but}}>> Q2 was disappointing.  
It’s a sturdy phone, <<although the weight {{although}}>> is a drawback.
"""

EXAMPLE_3_ACTIVATIONS = """
… {{however}} 2.031 : [(";" :2), ("however":8)]  
… {{but}}     1.774 : [("but":8)]  
… {{although}}1.423 : [("," :2), ("although":7)]
"""

EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
1. Rows  
   • however 2.031 ← semicolon 2 + however 8  
   • but     1.774 ← but 8  
   • although 1.423 ← comma 2 + although 7  

2. Pattern → adversative connectors; punctuation boosts weight when the marker is clause-initial.

3. Gradient → clause-initial discourse marker (however) > simple coordinating but > embedded concessive although.

4. Hypothesis → detector highlights contrastive links with bonus for turn-opening position.

5. Compress →
"""

EXAMPLE_3_EXPLANATION = """
[EXPLANATION]: Adversative conjunctions and discourse markers show a graded boost—strongest when launching a new clause (“however”), moderate for simple coordination (“but”), weakest when embedded (“although”).



"""

# 이렇게 두루뭉실한 예시 말고 실제로 바꿔서 해야할듯. 그리고 wait도 넣고.