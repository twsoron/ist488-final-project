def get_metadata(file_path):
    path = file_path.lower()

    if "course_info" in path:
        return {"type": "syllabus"}
    elif "hw_lab_code" in path:
        return {"type": "assignment"}
    elif "notes_lecture" in path:
        return {"type": "concept"}