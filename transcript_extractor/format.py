def format_transcript(transcript_segments):

    if not transcript_segments:
        return ""
    
    full_text = ""
    for segment in transcript_segments:
        text = segment.get('text', '')
        # Add a space between segments if needed
        if full_text and not full_text.endswith((' ', '\n')):
            full_text += " "
        full_text += text
    
    return full_text
