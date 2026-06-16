def content_to_string(content) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                text = b.get('text')
                type_ = b.get('type')
                if isinstance(text, str) and (type_ == 'text' or type_ is None):
                    parts.append(text)
        return ''.join(parts)
    return ''

def flatten_message_content(messages: list) -> list:
    flat = []
    for m in messages:
        m_copy = dict(m)
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            pass
        else:
            m_copy['content'] = content_to_string(m.get('content'))
        flat.append(m_copy)
    return flat

def content_has_image(content) -> bool:
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict):
            type_ = block.get('type')
            if type_ in ('image_url', 'image'):
                return True
    return False

def message_has_image(messages: list) -> bool:
    for m in messages:
        if content_has_image(m.get('content')):
            return True
    return False

def normalize_outbound_content(payload: dict) -> dict:
    choices = payload.get('choices')
    if not isinstance(choices, list):
        return payload
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get('delta')
        if isinstance(delta, dict) and isinstance(delta.get('content'), list):
            delta['content'] = content_to_string(delta['content'])
        
        message = choice.get('message')
        if isinstance(message, dict) and isinstance(message.get('content'), list):
            message['content'] = content_to_string(message['content'])
    return payload
