from mkmmd.cli.assets import extra_credits


def test_no_lines_no_section():
    assert extra_credits([]) == "" and extra_credits(None) == ""


def test_plain_lines_become_bullets_under_also():
    t = extra_credits(["Song: A Band", "Palette: Some Palette"])
    assert t == "\n## Also\n\n- Song: A Band\n- Palette: Some Palette\n"


def test_headings_pass_through_and_group_what_follows():
    t = extra_credits(["## Music", "A Song", "# Palette", "Some Palette", "", "  "])
    assert t.splitlines() == ["", "## Music", "", "- A Song", "", "## Palette", "", "- Some Palette"]


def test_text_before_the_first_heading_is_also():
    t = extra_credits(["loose line", "## Music", "a song"])
    assert "## Also" in t and t.index("## Also") < t.index("## Music")
