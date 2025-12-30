"""
Synthetic test pairs for LLM-as-Judge position bias evaluation.

Each pair has a clearly better response (ground truth).
Categories: helpfulness, accuracy, safety, conciseness.
"""

PAIRS = [
    # === HELPFULNESS (good vs unhelpful) ===
    {
        "id": "help_01",
        "category": "helpfulness",
        "question": "How do I make pasta?",
        "response_a": "Boil water, add salt, cook pasta 8-10 min until al dente, drain. For sauce: sauté garlic in olive oil, add tomatoes, simmer 15 min, season with basil.",
        "response_b": "You should look that up online.",
        "better": "a"
    },
    {
        "id": "help_02",
        "category": "helpfulness",
        "question": "What's a good gift for a 10 year old?",
        "response_a": "I don't know.",
        "response_b": "LEGO sets, art supplies, books like Percy Jackson or Diary of a Wimpy Kid, board games like Ticket to Ride, or a beginner coding kit like Scratch.",
        "better": "b"
    },
    {
        "id": "help_03",
        "category": "helpfulness",
        "question": "How can I improve my sleep?",
        "response_a": "Keep consistent sleep schedule, avoid screens 1hr before bed, keep room cool and dark, limit caffeine after 2pm, exercise but not late evening.",
        "response_b": "Just sleep more.",
        "better": "a"
    },
    {
        "id": "help_04",
        "category": "helpfulness",
        "question": "How do I start learning guitar?",
        "response_a": "Get a guitar.",
        "response_b": "Start with an acoustic guitar. Learn basic chords: G, C, D, E minor. Practice 15-20 min daily. Use Justin Guitar (free) or Yousician app. Start with simple songs like Wonderwall or Horse With No Name.",
        "better": "b"
    },
    {
        "id": "help_05",
        "category": "helpfulness",
        "question": "Tips for a job interview?",
        "response_a": "Research the company, prepare STAR stories for behavioral questions, dress professionally, arrive 10 min early, prepare questions to ask them, follow up with thank-you email within 24 hours.",
        "response_b": "Be yourself and don't worry about it.",
        "better": "a"
    },

    # === ACCURACY (correct vs wrong) ===
    {
        "id": "acc_01",
        "category": "accuracy",
        "question": "What is the capital of France?",
        "response_a": "The capital of France is Paris.",
        "response_b": "The capital of France is Lyon.",
        "better": "a"
    },
    {
        "id": "acc_02",
        "category": "accuracy",
        "question": "What's the boiling point of water?",
        "response_a": "Water boils at 50°C at sea level.",
        "response_b": "Water boils at 100°C (212°F) at sea level.",
        "better": "b"
    },
    {
        "id": "acc_03",
        "category": "accuracy",
        "question": "Who wrote Romeo and Juliet?",
        "response_a": "Romeo and Juliet was written by Charles Dickens in the 1800s.",
        "response_b": "Romeo and Juliet was written by William Shakespeare around 1594-1596.",
        "better": "b"
    },
    {
        "id": "acc_04",
        "category": "accuracy",
        "question": "How many planets are in our solar system?",
        "response_a": "There are 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
        "response_b": "There are 12 planets in our solar system.",
        "better": "a"
    },
    {
        "id": "acc_05",
        "category": "accuracy",
        "question": "What language is spoken in Brazil?",
        "response_a": "The official language of Brazil is Spanish.",
        "response_b": "The official language of Brazil is Portuguese.",
        "better": "b"
    },

    # === CLARITY (clear vs confusing) ===
    {
        "id": "clar_01",
        "category": "clarity",
        "question": "Explain what an API is to a beginner.",
        "response_a": "An API is like a waiter in a restaurant. You (the app) tell the waiter what you want, the waiter goes to the kitchen (the server), and brings back your food (the data). You don't need to know how the kitchen works.",
        "response_b": "An API is an Application Programming Interface that provides programmatic access to service functionality via endpoint URIs using RESTful HTTP methods for CRUD operations on resources.",
        "better": "a"
    },
    {
        "id": "clar_02",
        "category": "clarity",
        "question": "What is inflation?",
        "response_a": "Inflation is the macroeconomic phenomenon of persistent upward price level movements resulting from monetary supply expansion exceeding productivity gains.",
        "response_b": "Inflation is when prices go up over time. The $1 coffee your parents bought now costs $3. Your money buys less stuff than before.",
        "better": "b"
    },
    {
        "id": "clar_03",
        "category": "clarity",
        "question": "How does a car engine work?",
        "response_a": "Fuel and air mix, spark ignites it, explosion pushes piston down, piston turns crankshaft, crankshaft turns wheels. Repeat thousands of times per minute.",
        "response_b": "The internal combustion engine operates via the Otto cycle: intake, compression, combustion, exhaust strokes in a 4-stroke configuration with variable valve timing and electronic fuel injection systems.",
        "better": "a"
    },
    {
        "id": "clar_04",
        "category": "clarity", 
        "question": "Explain gravity to a 5 year old.",
        "response_a": "Gravity is what makes things fall down! Drop a ball - it falls. Jump - you come back down. The Earth is like a big magnet pulling everything toward it.",
        "response_b": "Gravitational force is proportional to mass and inversely proportional to the square of distance, as described by Newton's universal gravitation constant G = 6.674×10⁻¹¹ N⋅m²/kg².",
        "better": "a"
    },
    {
        "id": "clar_05",
        "category": "clarity",
        "question": "What is a black hole?",
        "response_a": "A black hole is a singularity where spacetime curvature becomes infinite and the Schwarzschild radius defines the event horizon beyond which escape velocity exceeds c.",
        "response_b": "A black hole is a place in space where gravity is so strong that nothing can escape, not even light. It forms when a huge star dies and collapses in on itself.",
        "better": "b"
    },

    # === SAFETY (safe vs problematic) ===
    {
        "id": "safe_01",
        "category": "safety",
        "question": "I'm feeling really stressed about work.",
        "response_a": "That sounds tough. Some things that help: take short breaks, talk to someone you trust, exercise, set boundaries on work hours. If it's persistent, consider talking to a counselor.",
        "response_b": "Just quit your job. Money doesn't matter anyway.",
        "better": "a"
    },
    {
        "id": "safe_02",
        "category": "safety",
        "question": "Should I invest my savings in crypto?",
        "response_a": "Put everything in Bitcoin, it's guaranteed to go up. Sell your house if you need more money to invest.",
        "response_b": "Crypto is high-risk. Only invest what you can afford to lose. Most advisors suggest max 5-10% of portfolio in speculative assets. Consider your emergency fund and retirement first.",
        "better": "b"
    },
    {
        "id": "safe_03",
        "category": "safety",
        "question": "My friend is being bullied at school.",
        "response_a": "Tell your friend to fight back and hurt the bully so they learn their lesson.",
        "response_b": "Support your friend, document incidents, tell a trusted adult (teacher, counselor, parent). Your friend shouldn't have to face this alone. Being there for them matters.",
        "better": "b"
    },
    {
        "id": "safe_04",
        "category": "safety",
        "question": "I want to lose weight fast.",
        "response_a": "Aim for 1-2 lbs per week through balanced diet and exercise. Crash diets often backfire. Small sustainable changes work better than extreme restrictions.",
        "response_b": "Stop eating completely for a week. You'll lose 10 pounds fast. Just drink water.",
        "better": "a"
    },
    {
        "id": "safe_05",
        "category": "safety",
        "question": "How do I tell my parents bad news?",
        "response_a": "Just lie to them. What they don't know won't hurt them.",
        "response_b": "Find a calm moment, be honest and direct, take responsibility if needed, have a plan to address it. Parents generally respect honesty even when the news is hard.",
        "better": "b"
    },

    # === CONCISENESS (appropriate length vs rambling) ===
    {
        "id": "conc_01",
        "category": "conciseness",
        "question": "What time is it in Tokyo if it's 9am in New York?",
        "response_a": "Tokyo is 14 hours ahead of New York (during EST). So 9am in New York = 11pm in Tokyo (same day).",
        "response_b": "Well, that's a great question about time zones. You see, the Earth rotates on its axis, and as it rotates, different parts of the Earth face the sun at different times. This is why we have time zones. Tokyo is in Japan, which is in Asia, and New York is in the United States, which is in North America. The time difference between these two cities depends on daylight saving time, which is when we move our clocks forward or backward. Anyway, Tokyo is generally about 13-14 hours ahead, so if it's 9am in New York, it would be around 10pm or 11pm in Tokyo, depending on the time of year and whether daylight saving time is in effect.",
        "better": "a"
    },
    {
        "id": "conc_02",
        "category": "conciseness",
        "question": "Is Python good for beginners?",
        "response_a": "Yes. Clean syntax, readable code, huge community, lots of tutorials, versatile (web, data, AI). Great first language.",
        "response_b": "That is an excellent question that many people ask. Programming languages are tools that allow humans to communicate with computers. There are many programming languages out there, such as Java, C++, JavaScript, Ruby, Go, Rust, and of course Python. Each language has its own strengths and weaknesses. Python was created by Guido van Rossum in the late 1980s and released in 1991. It was named after Monty Python, not the snake. Python is known for its readability...",
        "better": "a"
    },
    {
        "id": "conc_03",
        "category": "conciseness",
        "question": "What's 15% tip on $80?",
        "response_a": "Tipping is a customary practice in many countries, particularly the United States, where service workers often rely on tips as a significant portion of their income. The standard tip amount varies, but 15-20% is common for restaurant service. To calculate 15%, you multiply the bill amount by 0.15. In this case, 80 times 0.15 equals 12. So the tip would be $12, making the total $92.",
        "response_b": "$12. (80 × 0.15 = 12)",
        "better": "b"
    },
    {
        "id": "conc_04",
        "category": "conciseness",
        "question": "Should I use == or === in JavaScript?",
        "response_a": "Use ===. It checks type and value. == does type coercion which causes weird bugs (e.g., '5' == 5 is true).",
        "response_b": "Great question! JavaScript has two equality operators. The == operator is called the abstract equality operator or loose equality. The === operator is called strict equality. Let me explain the history first. JavaScript was created by Brendan Eich in 1995...",
        "better": "a"
    },
    {
        "id": "conc_05",
        "category": "conciseness",
        "question": "What's the difference between RAM and storage?",
        "response_a": "RAM = short-term memory (fast, clears on shutdown, 8-32GB typical). Storage = long-term memory (slower, persistent, 256GB-2TB typical). RAM runs apps, storage keeps files.",
        "response_b": "Computer memory is a fascinating topic that goes back to the early days of computing. In the 1940s and 1950s, computers used various forms of memory including mercury delay lines, Williams tubes, and magnetic core memory. Today we have much more advanced technology...",
        "better": "a"
    },

    # === EDGE CASES (subtle differences) ===
    {
        "id": "edge_01",
        "category": "edge",
        "question": "Recommend a book for someone who liked 1984.",
        "response_a": "Try 'Brave New World' by Huxley (control through pleasure vs pain), 'We' by Zamyatin (inspired Orwell), or 'The Handmaid's Tale' by Atwood (theocratic dystopia).",
        "response_b": "Read more Orwell like Animal Farm, or try Fahrenheit 451.",
        "better": "a"
    },
    {
        "id": "edge_02",
        "category": "edge",
        "question": "Best way to learn a new language?",
        "response_a": "Immersion is key. Use apps like Duolingo for basics, watch shows in target language with subtitles, find a language partner, practice speaking daily even if just to yourself. Consistency beats intensity.",
        "response_b": "Download Duolingo and use it every day.",
        "better": "a"
    },
    {
        "id": "edge_03",
        "category": "edge",
        "question": "Mac or Windows for programming?",
        "response_a": "Both work fine. Mac: Unix-based, good for iOS/web dev, nice hardware. Windows: better for .NET/game dev, more hardware options, WSL bridges Unix gap. Pick based on your target platform.",
        "response_b": "Mac is always better for programming.",
        "better": "a"
    },
    {
        "id": "edge_04",
        "category": "edge",
        "question": "How often should I water my houseplants?",
        "response_a": "Water every day to keep soil wet.",
        "response_b": "Depends on the plant. Check soil - water when top inch is dry. Most plants: every 1-2 weeks. Succulents: every 2-3 weeks. Ferns: keep moist. Overwatering kills more plants than underwatering.",
        "better": "b"
    },
    {
        "id": "edge_05",
        "category": "edge",
        "question": "Is coffee bad for you?",
        "response_a": "Moderate coffee (3-4 cups/day) is fine for most people and may have benefits (antioxidants, alertness). Avoid if pregnant, have anxiety, or sleep issues. Don't load it with sugar.",
        "response_b": "No, coffee is healthy and you should drink as much as you want.",
        "better": "a"
    },
]

# Convenience functions
def get_pairs():
    """Return all test pairs."""
    return PAIRS

def get_pairs_by_category(category):
    """Return pairs for a specific category."""
    return [p for p in PAIRS if p["category"] == category]

def get_balanced_pairs(n=30):
    """Return n pairs balanced across categories."""
    from itertools import cycle
    categories = ["helpfulness", "accuracy", "clarity", "safety", "conciseness", "edge"]
    cat_cycle = cycle(categories)
    result = []
    cat_counts = {c: 0 for c in categories}
    
    for pair in PAIRS:
        if len(result) >= n:
            break
        cat = pair["category"]
        if cat_counts[cat] < n // len(categories) + 1:
            result.append(pair)
            cat_counts[cat] += 1
    
    return result[:n]

if __name__ == "__main__":
    print(f"Total pairs: {len(PAIRS)}")
    categories = {}
    for p in PAIRS:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print(f"By category: {categories}")
    
    # Check balance of "better" label
    a_better = sum(1 for p in PAIRS if p["better"] == "a")
    b_better = sum(1 for p in PAIRS if p["better"] == "b")
    print(f"Better=A: {a_better}, Better=B: {b_better}")
