import json

def repair_tool_arguments(args: str, param_schema: dict = None) -> str:
    try:
        parsed = json.loads(args)
    except Exception:
        return args

    changed = False

    # Whole-arguments double encoding: '"{\\"a\\":1}"' parses to a string that is itself JSON of an object.
    if isinstance(parsed, str):
        try:
            inner = json.loads(parsed)
            if isinstance(inner, dict):
                parsed = inner
                changed = True
            else:
                return args
        except Exception:
            return args

    if not isinstance(parsed, dict):
        return json.dumps(parsed) if changed else args

    props = param_schema.get('properties') if param_schema else None
    if props and isinstance(props, dict):
        for key, value in list(parsed.items()):
            if not isinstance(value, str):
                continue
            prop_schema = props.get(key)
            if not isinstance(prop_schema, dict):
                continue
            want = prop_schema.get('type')
            if want not in ('array', 'object'):
                continue
            
            trimmed = value.strip()
            if not (trimmed.startswith('[') or trimmed.startswith('{')):
                continue
            try:
                inner = json.loads(trimmed)
                is_match = (
                    (want == 'array' and isinstance(inner, list)) or
                    (want == 'object' and isinstance(inner, dict))
                )
                if is_match:
                    parsed[key] = inner
                    changed = True
            except Exception:
                pass

    return json.dumps(parsed) if changed else args

def tool_schema_map(tools: list) -> dict:
    schema_map = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        func = t.get('function')
        if not isinstance(func, dict):
            continue
        name = func.get('name')
        params = func.get('parameters')
        if t.get('type') == 'function' and name and isinstance(params, dict):
            schema_map[name] = params
    return schema_map
