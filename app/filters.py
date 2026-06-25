BLOCKLIST = {
    "kill", "murder", "gore", "weapon", "blood", "torture", "massacre",
    "assassin", "genocide", "rape", "assault", "bomb", "terrorist",
    "nude", "naked", "sexual", "pornograph", "explicit",
    "mickey mouse", "darth vader", "harry potter", "batman", "superman",
    "spiderman", "iron man", "pikachu", "mario", "sonic",
    "suicide", "self-harm", "cutting",
}

def check_prompt(prompt: str) -> None:
    """Raise ValueError if prompt contains blocked content."""
    lower = prompt.lower()
    for term in BLOCKLIST:
        if term in lower:
            raise ValueError(
                "Your prompt contains restricted content. Please revise and try again."
            )
