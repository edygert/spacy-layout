# Differences Between Fork and Upstream

## Summary of Changes

This fork adds table row and cell bounding box extraction functionality to spacy-layout.

### Files Modified:
- `.gitignore` - Added project-specific files (CLAUDE.md, FINAL_SPECIFICATIONS.md, .obsidian/)
- `README.md` - Added comprehensive documentation for table bounding box features
- `pyproject.toml` - Added (new file) with project metadata and version 0.0.13
- `setup.cfg` - Updated version from 0.0.12 to 0.0.13
- `spacy_layout/layout.py` - Added table layout extraction functionality
- `spacy_layout/types.py` - Added TableLayout, TableRowLayout, and TableCellLayout dataclasses
- `spacy_layout/util.py` - Added serialization support for new table layout types
- `tests/test_general.py` - Added comprehensive tests for table layout features

### Key Features Added:
1. **Table Structure Extraction**
   - Extract bounding boxes for individual table cells
   - Extract bounding boxes for table rows
   - Support for cell properties (headers, spans, text content)

2. **New Data Structures**
   - `TableLayout` - Complete table structure information
   - `TableRowLayout` - Row-level bounding boxes and cells
   - `TableCellLayout` - Cell-level bounding boxes and metadata

3. **API Enhancements**
   - `Span._.table_layout` attribute for accessing table structure
   - Msgpack serialization support for table layout data
   - Empty row filtering and coordinate transformation utilities

### Commits:
See `commit_differences.txt` for the list of commits unique to this fork.

### Full Patch:
See `fork_differences.patch` for the complete diff that can be applied to upstream.

## Usage Example:

```python
from docling.datamodel.pipeline_options import PdfPipelineOptions
import spacy
from spacy_layout import spaCyLayout

# Enable table structure extraction
pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.do_cell_matching = True

nlp = spacy.blank("en")
layout = spaCyLayout(nlp, docling_options={"application/pdf": pipeline_options})
doc = layout("document.pdf")

# Access table layout
for table in doc._.tables:
    table_layout = table._.table_layout
    if table_layout:
        for row in table_layout.rows:
            print(f"Row {row.row_index}: {row.x}, {row.y}, {row.width}, {row.height}")
        for cell in table_layout.cells:
            print(f"Cell [{cell.row_index},{cell.col_index}]: {cell.text}")
```