# feature_engineering.py

def extract_features(doc):
    """Extracts a list of feature dictionaries for every line in the document."""
    features = []

    all_sizes = [span["size"] for page in doc for block in page.get_text("dict")["blocks"] for line in block.get("lines", []) for span in line.get("spans", [])]
    avg_doc_size = sum(all_sizes) / len(all_sizes) if all_sizes else 12

    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        blocks = page.get_text("dict")["blocks"]
        for i, block in enumerate(blocks):
            for j, line in enumerate(block.get("lines", [])):
                line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                if not line_text:
                    continue

                sizes = [span["size"] for span in line["spans"]]
                avg_size = sum(sizes) / len(sizes) if sizes else 0
                is_bold = any(span["flags"] & 2**4 for span in line["spans"])
                is_all_caps = line_text.isupper() and len(line_text) > 1

                line_bbox = line['bbox']
                line_center = (line_bbox[0] + line_bbox[2]) / 2
                page_center = page_width / 2
                is_centered = abs(line_center - page_center) < (page_width * 0.1)

                space_below = 0
                if j + 1 < len(block.get("lines", [])):
                    next_line_bbox = block.get("lines", [])[j+1]['bbox']
                    space_below = next_line_bbox[1] - line_bbox[3]
                elif i + 1 < len(blocks):
                    next_block_bbox = blocks[i+1]['bbox']
                    space_below = next_block_bbox[1] - line_bbox[3]

                features.append({
                    "text": line_text, "page_num": page_num,
                    "avg_size": avg_size, "is_bold": is_bold,
                    "word_count": len(line_text.split()),
                    "relative_size": avg_size / avg_doc_size if avg_doc_size > 0 else 1,
                    "x_pos": line_bbox[0],
                    "is_all_caps": is_all_caps,
                    "is_centered": is_centered,
                    "space_below": space_below,
                })
    return features