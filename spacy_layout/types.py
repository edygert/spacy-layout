from dataclasses import dataclass

from docling_core.types.doc.document import (
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

DoclingItem = ListItem | SectionHeaderItem | TextItem | TableItem


@dataclass
class Attrs:
    """Custom atributes used to extend spaCy"""

    doc_layout: str
    doc_pages: str
    doc_tables: str
    doc_markdown: str
    span_layout: str
    span_data: str
    span_heading: str
    span_group: str
    span_table_layout: str = "table_layout"


@dataclass
class PageLayout:
    page_no: int
    width: float
    height: float

    @classmethod
    def from_dict(cls, data: dict) -> "PageLayout":
        return cls(**data)


@dataclass
class DocLayout:
    """Document layout features added to Doc object"""

    pages: list[PageLayout]

    @classmethod
    def from_dict(cls, data: dict) -> "DocLayout":
        pages = [PageLayout.from_dict(page) for page in data.get("pages", [])]
        return cls(pages=pages)


@dataclass
class SpanLayout:
    """Text span layout features added to Span object"""

    x: float
    y: float
    width: float
    height: float
    page_no: int

    @classmethod
    def from_dict(cls, data: dict) -> "SpanLayout":
        return cls(**data)


@dataclass
class TableCellLayout:
    """Layout information for a table cell"""

    x: float
    y: float
    width: float
    height: float
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1
    is_column_header: bool = False
    is_row_header: bool = False
    text: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "TableCellLayout":
        return cls(**data)


@dataclass
class TableRowLayout:
    """Layout information for a table row"""

    x: float
    y: float
    width: float
    height: float
    row_index: int
    cells: list[TableCellLayout]

    @classmethod
    def from_dict(cls, data: dict) -> "TableRowLayout":
        cells = [TableCellLayout.from_dict(cell) for cell in data.get("cells", [])]
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            row_index=data["row_index"],
            cells=cells,
        )


@dataclass
class TableLayout:
    """Complete layout information for a table"""

    rows: list[TableRowLayout]
    cells: list[TableCellLayout]
    page_no: int

    @classmethod
    def from_dict(cls, data: dict) -> "TableLayout":
        rows = [TableRowLayout.from_dict(row) for row in data.get("rows", [])]
        cells = [TableCellLayout.from_dict(cell) for cell in data.get("cells", [])]
        return cls(rows=rows, cells=cells, page_no=data.get("page_no", 0))
