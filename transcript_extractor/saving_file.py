def save_transcript_to_file(transcript, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        return True
    except Exception as e:
        print(f"Error saving transcript: {e}")
        return False