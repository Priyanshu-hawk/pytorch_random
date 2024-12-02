import random
import string

def generate_sorry_variations(count=670):
    base_word = "sorry"
    variations = set()
    
    # Possible prefixes and suffixes
    prefixes = ["so", "very", "mega", "ultra", "super", "hyper", "extra", "woops-", "oh-", "uh-"]
    suffixes = ["!", "...", "?", "😔", "😭", "🙁", "-pls", "-forgive", "-oops", "-mybad", "🥺"]
    styles = ["UPPER", "lower", "Title", "Camel"]
    
    while len(variations) < count:
        # Add prefix, suffix, or random characters
        prefix = random.choice(prefixes) if random.random() < 0.5 else ""
        suffix = random.choice(suffixes) if random.random() < 0.5 else ""
        
        # Apply random casing
        style = random.choice(styles)
        if style == "UPPER":
            sorry = base_word.upper()
        elif style == "lower":
            sorry = base_word.lower()
        elif style == "Title":
            sorry = base_word.title()
        elif style == "Camel":
            sorry = base_word[0].lower() + base_word[1:].upper()
        
        # Add random decoration
        # print(string.punctuation)
        decoration = "".join(random.choices(["!!!","..."], k=random.randint(1,1)))
        variation = f"{prefix}{sorry}{suffix}{decoration}"
        variations.add(variation)
    
    return list(variations)

# Number of variations to generate
num_variations = 670

# Generate and print variations
sorry_variations = generate_sorry_variations(num_variations)
for i, variation in enumerate(sorry_variations, 1):
    print(f"{i}: {variation}")
