import re
import json
from typing import Union, Optional, Set

DIALECT_MARKERS = [
    '<|tool_calls_section_begin|>',
    '<|tool_call_begin|>',
    '<tool_call>',
    '<function=',
]

def starts_with_dialect_marker(text: str) -> bool:
    t = text.lstrip()
    return any(t.startswith(m) for m in DIALECT_MARKERS)

def could_become_dialect_marker(text: str) -> bool:
    t = text.lstrip()
    if not t:
        return True
    return any(m.startswith(t) and len(t) < len(m) for m in DIALECT_MARKERS)

def contains_dialect_marker(text: str) -> bool:
    return any(m in text for m in DIALECT_MARKERS)



def extract_balanced_json(text: str, start_index: int) -> Optional[dict]:
    if start_index >= len(text):
        return None
    open_char = text[start_index]
    if open_char not in ('{', '['):
        return None
    close_char = '}' if open_char == '{' else ']'
    depth = 0
    in_string = False
    escaped = False
    for i in range(start_index, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return {'json': text[start_index:i + 1], 'end': i + 1}
    return None

def is_known_tool(name: str, tool_names: set) -> bool:
    return len(tool_names) == 0 or name in tool_names

def call_from_named_json(json_str: str, tool_names: set) -> Optional[dict]:
    try:
        obj = json.loads(json_str)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get('name')
    if not isinstance(name, str) or not name or not is_known_tool(name, tool_names):
        return None
    raw_args = obj.get('arguments') or obj.get('parameters') or {}
    args = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
    try:
        json.loads(args)
    except Exception:
        return None
    return {'name': name, 'arguments': args}

def parse_token_dialect(text: str, tool_names: set) -> dict:
    calls = []
    clean = text
    clean = clean.replace('<|tool_calls_section_begin|>', '').replace('<|tool_calls_section_end|>', '')
    
    pattern = re.compile(r'<\|tool_call_begin\|>\s*([\s\S]*?)\s*<\|tool_call_argument_begin\|>\s*')
    spans = []
    parsed_all = True
    
    for m in pattern.finditer(clean):
        id_token = m.group(1).strip()
        arg_start = m.end()
        json_start = clean.find('{', arg_start)
        extracted = None
        if json_start != -1:
            extracted = extract_balanced_json(clean, json_start)
            
        name_match = re.match(r'^functions\.([A-Za-z0-9_.-]+):\d+$', id_token)
        name = name_match.group(1) if name_match else None
        
        args_ok = False
        if extracted and name and is_known_tool(name, tool_names):
            try:
                json.loads(extracted['json'])
                args_ok = True
            except Exception:
                pass
            if args_ok:
                calls.append({'name': name, 'arguments': extracted['json']})
                
        if not args_ok:
            parsed_all = False
            
        end_search_from = extracted['end'] if extracted else arg_start
        end_tag_pos = clean.find('<|tool_call_end|>', end_search_from)
        
        to_idx = end_tag_pos + len('<|tool_call_end|>') if end_tag_pos != -1 else (extracted['end'] if extracted else arg_start)
        spans.append((m.start(), to_idx))
        
    for start, end in reversed(spans):
        clean = clean[:start] + clean[end:]
        
    return {'calls': calls if (parsed_all and len(calls) > 0) else None, 'cleanText': clean.strip()}

def parse_function_tag_dialect(text: str, tool_names: set) -> dict:
    calls = []
    clean = text
    parsed_all = True
    
    pattern = re.compile(r'<function=([A-Za-z0-9_.-]+)\s*>?\s*')
    spans = []
    
    for m in pattern.finditer(text):
        name = m.group(1)
        after_head = m.end()
        json_start = after_head if after_head < len(text) and text[after_head] in ('{', '[') else text.find('{', after_head)
        extracted = None
        if json_start != -1:
            extracted = extract_balanced_json(text, json_start)
            
        ok = False
        if extracted and is_known_tool(name, tool_names) and extracted['json'].startswith('{'):
            try:
                json.loads(extracted['json'])
                ok = True
            except Exception:
                pass
            if ok:
                calls.append({'name': name, 'arguments': extracted['json']})
                
        if not ok:
            parsed_all = False
            
        close_search_from = extracted['end'] if extracted else after_head
        close_tag_pos = text.find('</function>', close_search_from)
        to_idx = close_tag_pos + len('</function>') if close_tag_pos != -1 else close_search_from
        spans.append((m.start(), to_idx))
        
    for start, end in reversed(spans):
        clean = clean[:start] + clean[end:]
        
    return {'calls': calls if (parsed_all and len(calls) > 0) else None, 'cleanText': clean.strip()}

def parse_xml_dialect(text: str, tool_names: set) -> dict:
    calls = []
    parsed_all = True
    pattern = re.compile(r'<tool_call>\s*([\s\S]*?)\s*</tool_call>')
    clean = text
    matches = pattern.findall(text)
    for inner in matches:
        call = call_from_named_json(inner, tool_names)
        if call:
            calls.append(call)
        else:
            parsed_all = False
            
    clean = pattern.sub('', clean)
    if '<tool_call>' in clean:
        parsed_all = False
        clean = re.sub(r'<tool_call>[\s\S]*$', '', clean)
        
    return {'calls': calls if (parsed_all and len(calls) > 0) else None, 'cleanText': clean.strip()}

def rescue_inline_tool_calls(text: str, tool_names: set) -> dict:
    if not text:
        return {'detected': False, 'calls': None, 'cleanText': text}
        
    if '<|tool_call_begin|>' in text or '<|tool_calls_section_begin|>' in text:
        res = parse_token_dialect(text, tool_names)
        return {'detected': True, 'calls': res['calls'], 'cleanText': res['cleanText']}
        
    if '<function=' in text:
        res = parse_function_tag_dialect(text, tool_names)
        return {'detected': True, 'calls': res['calls'], 'cleanText': res['cleanText']}
        
    if '<tool_call>' in text:
        res = parse_xml_dialect(text, tool_names)
        return {'detected': True, 'calls': res['calls'], 'cleanText': res['cleanText']}
        
    # Entire answer is one JSON object naming a known tool (bare or inside ```json)
    trimmed = text.strip()
    fenced_match = re.match(r'^```(?:json)?\s*([\s\S]*?)\s*```$', trimmed)
    candidate = (fenced_match.group(1) if fenced_match else trimmed).strip()
    
    if candidate.startswith('{') and candidate.endswith('}'):
        call = call_from_named_json(candidate, tool_names)
        if call:
            return {'detected': True, 'calls': [call], 'cleanText': ''}
            
    return {'detected': False, 'calls': None, 'cleanText': text}
