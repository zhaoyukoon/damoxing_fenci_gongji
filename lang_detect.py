import re
import fasttext
from loguru import logger

# Load fasttext model
lang_model = fasttext.load_model('lid.176.ftz')

# Regex patterns
chinese_pattern = re.compile(r'^[\u4E00-\u9FFF]+$')
english_pattern = re.compile(r'^[a-zA-Z]+$')
digit_pattern = re.compile(r'^[0-9]+$')
code_camel = re.compile('[a-z]+[A-Z][a-z]')
code_pattern = re.compile(r'^[.$>:][a-zA-Z]+')

def get_language_names():
    """
    Returns a dictionary mapping ISO 639-1 codes to their corresponding language names in English.
    """
    return {
        # Major world languages
        "en": "English",
        "zh": "Chinese",
        "es": "Spanish",
        "hi": "Hindi",
        "ar": "Arabic",
        "bn": "Bengali",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "de": "German",
        
        # European languages
        "fr": "French",
        "it": "Italian",
        "nl": "Dutch",
        "pl": "Polish",
        "el": "Greek",
        "cs": "Czech",
        "sv": "Swedish",
        "da": "Danish",
        "fi": "Finnish",
        "no": "Norwegian",
        
        # Asian languages
        "ko": "Korean",
        "vi": "Vietnamese",
        "th": "Thai",
        "tr": "Turkish",
        "fa": "Persian",
        "id": "Indonesian",
        "ms": "Malay",
        
        # Other major languages
        "he": "Hebrew",
        "uk": "Ukrainian",
        "ro": "Romanian",
        "hu": "Hungarian",
        "bg": "Bulgarian",
        "hr": "Croatian",
        "sr": "Serbian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "et": "Estonian",
        "lv": "Latvian",
        "lt": "Lithuanian"
    }

def get_language_name(iso_code):
    """
    Get the language name for a given ISO 639-1 code.
    """
    names = get_language_names()
    normalized_code = iso_code.lower().strip()
    return names.get(normalized_code) if normalized_code in names else 'NULL'

def detect_language(text):
    """
    Detect the probable language of a text string based on character ranges.
    Returns a dictionary with detected languages and their percentage of characters.
    """
    if not text:
        return {"unknown": 100.0}
    
    ranges = {
        "english": lambda c: ord('a') <= ord(c.lower()) <= ord('z'),
        "chinese": lambda c: 0x4E00 <= ord(c) <= 0x9FFF,
        "japanese": lambda c: (0x3040 <= ord(c) <= 0x309F) or
                            (0x30A0 <= ord(c) <= 0x30FF) or
                            (0x4E00 <= ord(c) <= 0x9FFF),
        "korean": lambda c: 0xAC00 <= ord(c) <= 0xD7AF,
        "russian": lambda c: 0x0400 <= ord(c) <= 0x04FF,
        "greek": lambda c: 0x0370 <= ord(c) <= 0x03FF,
        "arabic": lambda c: 0x0600 <= ord(c) <= 0x06FF,
        "hebrew": lambda c: 0x0590 <= ord(c) <= 0x05FF,
        "thai": lambda c: 0x0E00 <= ord(c) <= 0x0E7F
    }
    
    counts = {lang: 0 for lang in ranges}
    total_chars = 0
    
    for char in text:
        if char.isspace():
            continue
        total_chars += 1
        for lang, char_range in ranges.items():
            if char_range(char):
                counts[lang] += 1
                
    if total_chars == 0:
        return {"unknown": 100.0}
    
    percentages = {
        lang: (count / total_chars) * 100
        for lang, count in counts.items()
        if count > 0
    }
    
    return percentages if percentages else {"unknown": 100.0}

def get_primary_language(text):
    """
    Returns the most probable language for the given text.
    """
    try:
        langs = detect_language(text)
    except TypeError:
        logger.error(f'detect {text} failed')
        return 'NULL'
    if not langs or langs.get("unknown", 0) == 100.0:
        return "NULL"
    return max(langs.items(), key=lambda x: x[1])[0]

def refine_english_lang(s, pure=False):
    """
    Refine language detection for English text.
    """
    s = s.replace('▁', '').strip()
    if code_camel.match(s) or code_pattern.match(s):
        return 'code'
    lang = lang_model.predict(s, k=1)
    lang = lang[0][0].replace('__label__', '')

    if len(s) > 8 and lang.lower() == 'vietnamese':
        return 'english'

    lang = get_language_name(lang).lower()
    if lang == 'null':
        return 'english' if pure else 'NULL'
    return lang

def detect_lang(s):
    """
    Main language detection function.
    """
    if '\t' in s or '\\t' in s or '\n' in s or '\\n' in s:
        return 'control'
    if english_pattern.match(s):
        return refine_english_lang(s, True)
    if chinese_pattern.match(s):
        return 'chinese'
    if digit_pattern.match(s):
        return 'digits'
    lang = get_primary_language(s)
    if lang == 'english':
        return refine_english_lang(s)

    if lang == 'NULL': 
        if code_pattern.match(s) or code_camel.match(s):
            return 'code'
    return lang
