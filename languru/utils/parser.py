import re
from typing import List, Text


def parse_tag_content(text: Text, tag: Text) -> List[Text]:
    # Regular expression to match the content within the specified tag
    # This regex accounts for optional attributes within the tag
    pattern = rf"<{tag}(\s+[^>]*)?>(.*?)</{tag}>"

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Extract the content part from each match
    contents = [match[1].strip() for match in matches]

    return contents
