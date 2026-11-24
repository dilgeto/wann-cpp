"""Caption drawn inside a supplementary figure (bold title line + wrapped paragraph at the bottom)."""
import textwrap

FONT = 7.0
LINE_H = FONT * 1.28 / 72          # inches per line
CHAR_W = 0.45                      # average glyph width / font size (Liberation Sans); wrap width estimate


def wrap(text, width_in, fontsize=FONT):
    n = int((width_in - 0.3) * 72 / (fontsize * CHAR_W))
    return textwrap.wrap(text, n)


def caption_height(title, text, width_in):
    """Height in inches the caption needs at the bottom of a figure of the given width."""
    return (1 + len(wrap(text, width_in))) * LINE_H + 0.22


def draw_caption(fig, title, text):
    w, h = fig.get_size_inches()
    lines = wrap(text, w)
    y0 = 0.08 / h
    fig.text(0.015, y0 + len(lines) * LINE_H / h + 0.01 / h, title, fontsize=FONT + 0.5, fontweight="bold", va="bottom", ha="left")
    fig.text(0.015, y0, "\n".join(lines), fontsize=FONT, va="bottom", ha="left", linespacing=1.28)
