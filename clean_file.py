#!/usr/bin/env python3
"""Clean special characters from the enhanced_agri_safety_llm.py file."""

with open('enhanced_agri_safety_llm.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace special characters
replacements = {
    '•': '-',
    '✅': '*',
    '🔧': '-',
    '🌪️': '-',
    '⚡': '-',
    '🛡️': '-',
    '🎯': '-',
    '🏆': '-',
    '🚜': '-',
    '📊': '-',
    '🔍': '-',
    '🚜': 'AgriAI',
    '📊': 'Stats',
    '🔍': 'Analysis',
    '⚡': 'Performance',
    '🛡️': 'Safety',
    '🎯': 'Target',
    '🏆': 'Achievement'
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('enhanced_agri_safety_llm.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File cleaned successfully!")