import re
import pathlib

root = pathlib.Path(r'C:\Users\hs735.COLTSMOKE\OneDrive\Documents\Hackathon\agri_safety_ai')

# More comprehensive emoji regex
emoji_re = re.compile(
    r'[\U0001F600-\U0001F64F'  # emoticons
    r'\U0001F300-\U0001F5FF'  # symbols & pictographs
    r'\U0001F680-\U0001F6FF'  # transport & map symbols
    r'\U0001F1E0-\U0001F1FF'  # flags (iOS)
    r'\U00002500-\U00002BEF'  # chinese char
    r'\U00002702-\U000027B0'
    r'\U00002702-\U000027B0'
    r'\U000024C2-\U0001F251'
    r'\U0001f926-\U0001f937'
    r'\U00010000-\U0010ffff'
    r'\u2640-\u2642'
    r'\u2600-\u2B55'
    r'\u200d'
    r'\u23cf'
    r'\u23e9'
    r'\u231a'
    r'\ufe0f'  # dingbats
    r'\u3030'
    r']+', flags=re.UNICODE)

text_exts = {'.py', '.md', '.txt', '.json', '.yml', '.yaml', '.csv', '.ini', '.cfg', '.mdown', '.rst'}
modified = []

for path in root.rglob('*'):
    if path.is_file() and path.suffix.lower() in text_exts:
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        new_text = emoji_re.sub('', text)
        if new_text != text:
            path.write_text(new_text, encoding='utf-8')
            count = len(emoji_re.findall(text))
            modified.append((path.relative_to(root), count))

print('Modified files:', len(modified))
for p, c in modified:
    print(f'{p}: {c}')
