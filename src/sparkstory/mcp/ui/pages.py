"""HTML for the three pages, as module constants.

Not Jinja2: a template engine is a runtime dependency for three pages, against a
dependency count the README quotes. Revisit past three pages.

Everything here is a pure function from data to a string, so the properties worth
pinning -- no preselected pronouns, every beat rendered, no player for an absent
file -- are testable without an HTTP request.

**Escaping.** Every value that reaches HTML goes through ``html.escape``. A
premise and a child's name are free text typed by a person, and a story's prose
is written by a model; none of it may become markup.
"""

from html import escape
from pathlib import Path

from sparkstory.entities.stories import Story, StoryOutline
from sparkstory.mcp.ui.jobs import Job, JobState

_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  max-width: 46rem; margin: 0 auto; padding: 2rem 1rem; line-height: 1.6;
}
h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
.subtitle { opacity: 0.7; margin-top: 0; }
fieldset { border: 1px solid rgba(128,128,128,0.4); border-radius: 8px;
  padding: 1rem; margin-bottom: 1rem; }
legend { padding: 0 0.4rem; font-weight: 600; }
label { display: block; margin-bottom: 0.75rem; }
label > span { display: block; margin-bottom: 0.25rem; font-size: 0.9rem; }
input[type=text], input[type=number], select, textarea {
  width: 100%; padding: 0.5rem; border-radius: 6px;
  border: 1px solid rgba(128,128,128,0.5); background: transparent;
  color: inherit; font: inherit;
}
.radio-row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;
  margin-bottom: 0.75rem; }
.radio-row label { display: flex; align-items: center; gap: 0.35rem; margin: 0; }
button { padding: 0.7rem 1.4rem; border-radius: 6px; border: none;
  background: #4a90d9; color: white; font-size: 1rem; cursor: pointer; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.error { border: 1px solid #dc2626; background: rgba(220,38,38,0.08);
  padding: 0.75rem 1rem; border-radius: 6px; margin: 1rem 0; }
.beat { border-left: 3px solid rgba(128,128,128,0.4); padding-left: 0.9rem;
  margin-bottom: 1rem; }
.page { margin: 2rem 0; padding-bottom: 1.5rem;
  border-bottom: 1px solid rgba(128,128,128,0.2); }
.page img { max-width: 100%; border-radius: 8px; }
.page audio, .page video { width: 100%; margin-top: 0.5rem; }
.status { opacity: 0.8; font-style: italic; }
.books { list-style: none; padding: 0; }
.book { padding: 0.9rem 0; border-bottom: 1px solid rgba(128,128,128,0.2); }
.book a { text-decoration: none; font-size: 1.05rem; }
.book .meta { font-size: 0.85rem; opacity: 0.7; margin-top: 0.2rem; }
.badge { display: inline-block; margin-left: 0.4rem; padding: 0.05rem 0.45rem;
  border-radius: 999px; font-size: 0.75rem;
  border: 1px solid rgba(128,128,128,0.45); }
nav { margin-bottom: 1.5rem; font-size: 0.9rem; }
"""


def _document(title: str, body: str) -> str:
    """Wrap body content in a complete HTML document."""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"<title>{escape(title)}</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )


_FORM_BODY = """
<nav><a href="/library">Books you have already made &rarr;</a></nav>

<h1>Make a storybook</h1>
<p class="subtitle">Tell us about the child and the story you have in mind.</p>

<div class="error" id="error" hidden></div>

<form id="brief">
  <fieldset>
    <legend>The child</legend>
    <label><span>Their name</span>
      <input type="text" name="name" maxlength="40" required></label>
    <label><span>Their age</span>
      <input type="number" name="age" min="2" max="12" value="5" required></label>
    <div class="radio-row" role="radiogroup" aria-label="Pronouns">
      <strong>Pronouns</strong>
      <label><input type="radio" name="pronouns" value="she/her" required>
        she/her</label>
      <label><input type="radio" name="pronouns" value="he/him" required>
        he/him</label>
      <label><input type="radio" name="pronouns" value="they/them" required>
        they/them</label>
    </div>
    <label><span>Reading level</span>
      <select name="reading_level">
        <option value="pre_reader">Pre-reader (read aloud, ages 2-4)</option>
        <option value="early_reader" selected>Early reader (ages 4-6)</option>
        <option value="developing">Developing (ages 6-8)</option>
        <option value="confident">Confident (ages 8-10)</option>
      </select></label>
    <label><span>Things they love (comma separated)</span>
      <input type="text" name="interests" placeholder="foxes, the moon"></label>
  </fieldset>

  <fieldset>
    <legend>The story</legend>
    <label><span>What is it about?</span>
      <textarea name="premise" rows="3" minlength="3" maxlength="500" required
        placeholder="a fox who wants to visit the moon"></textarea></label>
    <label><span>Tone</span>
      <select name="tone">
        <option value="gentle" selected>Gentle</option>
        <option value="funny">Funny</option>
        <option value="adventurous">Adventurous</option>
        <option value="magical">Magical</option>
        <option value="heartwarming">Heartwarming</option>
      </select></label>
    <div class="radio-row" role="radiogroup" aria-label="World">
      <strong>World</strong>
      <label><input type="radio" name="world_rules" value="imaginative" checked>
        Imaginative</label>
      <label><input type="radio" name="world_rules" value="realistic">
        Realistic</label>
    </div>
    <label><span>How many pages?</span>
      <input type="number" name="page_count" min="4" max="24" value="6"></label>
    <label><span>Must include (comma separated)</span>
      <input type="text" name="must_include" placeholder="a paper rocket"></label>
    <label><span>Keep out of the story (comma separated)</span>
      <input type="text" name="avoid" placeholder="spiders, the dark"></label>
  </fieldset>

  <button type="submit" id="submit">Plan the story</button>
</form>

<script>
const form = document.getElementById('brief');
const errorBox = document.getElementById('error');
const submit = document.getElementById('submit');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  submit.disabled = true;
  submit.textContent = 'Planning...';

  const data = Object.fromEntries(new FormData(form).entries());
  try {
    const response = await fetch('/plan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data),
    });
    const body = await response.json();
    if (response.ok) {
      window.location = '/job/' + body.job_id;
    } else {
      errorBox.textContent = body.error || 'Something went wrong.';
      errorBox.hidden = false;
    }
  } catch (err) {
    errorBox.textContent = 'Could not reach the server: ' + err.message;
    errorBox.hidden = false;
  } finally {
    submit.disabled = false;
    submit.textContent = 'Plan the story';
  }
});
</script>
"""


def render_form() -> str:
    """The brief form -- the front door.

    **No pronoun option is preselected**, and all three are ``required``, so the
    parent must choose. ``Pronouns`` defaults to ``they/them`` in the schema,
    which is the right default for a program and the wrong one for a form: a
    preselected value is a guess a parent may not notice they are accepting.
    """
    return _document("Make a storybook - SparkStory", _FORM_BODY)


def render_library(books: list[dict[str, object]]) -> str:
    """Every finished book on disk, as a list a parent can browse.

    Shows the story's own title and the date, never the run directory name -- a
    directory is named after the premise, which is a parent's words about their
    child. See `ui/library.py` for the full argument.
    """
    if not books:
        body = (
            "<h1>No books yet</h1>"
            '<p>Books you make will appear here. <a href="/">Make one</a>.</p>'
        )
        return _document("Library - SparkStory", body)

    rows = []
    for book in books:
        badges = "".join(
            f'<span class="badge">{label}</span>'
            for key, label in (
                ("has_images", "pictures"),
                ("has_audio", "audio"),
                ("has_video", "video"),
                ("has_pdf", "PDF"),
            )
            if book.get(key)
        )
        rows.append(
            '<li class="book">'
            f'<a href="/library/{escape(str(book["run_id"]))}">'
            f"<strong>{escape(str(book['title']))}</strong></a>"
            f'<div class="meta">{escape(str(book["made"]))} &middot; '
            f"{book['pages']} pages {badges}</div>"
            "</li>"
        )

    body = (
        "<h1>Library</h1>"
        f'<p class="subtitle">{len(books)} books on this machine.</p>'
        f'<ul class="books">{"".join(rows)}</ul>'
        '<p><a href="/">Make another</a></p>'
    )
    return _document("Library - SparkStory", body)


def render_job(job: Job) -> str:
    """The job page: progress while working, the outline when it is ready.

    One page for four states rather than four pages, because the browser polls
    this same URL throughout and a redirect on every transition would fight the
    back button.
    """
    if job.state is JobState.FAILED:
        body = (
            "<h1>The story could not be planned</h1>"
            f'<div class="error">{escape(job.error or "Unknown error")}</div>'
            '<p><a href="/">Start again</a></p>'
        )
        return _document("Failed - SparkStory", body)

    if job.state is JobState.AWAITING_APPROVAL and job.outline is not None:
        return _document(
            "Approve the plan - SparkStory", _outline_body(job, job.outline)
        )

    if job.state is JobState.COMPLETE:
        body = (
            "<h1>Your book is ready</h1>"
            f'<p><a href="/job/{escape(job.id)}/book">Read it</a></p>'
        )
        return _document("Ready - SparkStory", body)

    return _document("Working - SparkStory", _progress_body(job))


def _progress_body(job: Job) -> str:
    """Shown while planning or writing. Polls until the state changes."""
    heading = (
        "Planning the story" if job.state is JobState.PLANNING else "Writing the book"
    )
    note = (
        "This takes a few minutes."
        if job.state is JobState.WRITING
        else "This usually takes under a minute."
    )
    return f"""
<h1>{escape(heading)}</h1>
<p class="status" id="detail">{escape(job.detail or note)}</p>
<script>
setInterval(async () => {{
  const response = await fetch('/job/{escape(job.id)}/status');
  if (!response.ok) return;
  const body = await response.json();
  const detail = document.getElementById('detail');
  if (body.detail) detail.textContent = body.detail;
  if (body.state !== '{escape(job.state.value)}') window.location.reload();
}}, 2000);
</script>
"""


def _outline_body(job: Job, outline: StoryOutline) -> str:
    """The whole outline, uncompressed, with approve and revise."""
    characters = "".join(
        f"<li><strong>{escape(character.name)}</strong> &mdash; "
        f"{escape(character.description)}</li>"
        for character in outline.characters
    )
    beats = "".join(
        f'<div class="beat"><strong>{beat.position}. {escape(beat.title)}</strong>'
        f" <em>({escape(beat.function.value.replace('_', ' '))})</em>"
        f"<br>{escape(beat.summary)}</div>"
        for beat in outline.beats
    )
    return f"""
<h1>{escape(outline.title)}</h1>
<p class="subtitle">{escape(outline.logline)}</p>
<p><strong>Theme:</strong> {escape(outline.theme)}</p>

<h2>Who is in it</h2>
<ul>{characters}</ul>

<h2>What happens</h2>
{beats}

<div class="error" id="error" hidden></div>

<p>
  <button id="approve">Write this book</button>
</p>

<details>
  <summary>Ask for a different plan</summary>
  <label><span>What would you change?</span>
    <textarea id="note" rows="2"
      placeholder="Make Maryam the one who wants to go"></textarea></label>
  <button id="revise">Plan it again</button>
</details>

<script>
const errorBox = document.getElementById('error');

async function post(path, payload) {{
  errorBox.hidden = true;
  const response = await fetch(path, {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(payload || {{}}),
  }});
  if (response.ok) {{
    window.location.reload();
  }} else {{
    const body = await response.json();
    errorBox.textContent = body.error || 'Something went wrong.';
    errorBox.hidden = false;
  }}
}}

document.getElementById('approve').addEventListener('click', () =>
  post('/job/{escape(job.id)}/approve'));
document.getElementById('revise').addEventListener('click', () =>
  post('/job/{escape(job.id)}/revise',
       {{note: document.getElementById('note').value}}));
</script>
"""


def render_book(job: Job, media: dict[str, object]) -> str:
    """The finished book for a job, with whatever media its directory holds."""
    if job.story is None or job.outline is None:  # pragma: no cover - handler guards
        return _document("Not ready - SparkStory", "<h1>This book is not ready</h1>")

    return _book_page(
        story=job.story,
        media=media,
        file_base=f"/job/{job.id}/file",
        run_directory=job.run_directory,
        back_link=("/", "Make another"),
    )


def render_library_book(
    story: Story, media: dict[str, object], run_id: str, run_directory: Path
) -> str:
    """The same book page, for a run read off disk rather than from a job.

    Shares ``_book_page`` with the job route rather than duplicating it: a second
    renderer would drift, and the property that matters -- no player for a file that
    is not there -- would then only be tested on one of them.
    """
    return _book_page(
        story=story,
        media=media,
        file_base=f"/library/{run_id}/file",
        run_directory=run_directory,
        back_link=("/library", "Back to the library"),
    )


def _book_page(
    *,
    story: Story,
    media: dict[str, object],
    file_base: str,
    run_directory: Path | None,
    back_link: tuple[str, str],
) -> str:
    """The finished book, with whatever media a run left in its directory.

    Nothing here generates media. Absent artifacts render as nothing at all --
    no placeholder, no disabled player -- so the page never implies a file exists
    that does not.
    """
    outline = story.outline
    raw_pages = media.get("pages", [])
    by_number = {
        int(page["number"]): page  # type: ignore[index,call-overload,arg-type]
        for page in raw_pages  # type: ignore[union-attr]
    }

    sections: list[str] = []
    for page in story.pages:
        parts = [f'<div class="page"><p>{escape(page.text)}</p>']
        assets = by_number.get(page.page_number, {})
        image = assets.get("image")
        audio = assets.get("audio")
        if image:
            parts.append(
                f'<img src="{file_base}/{escape(str(image))}" '
                f'alt="Illustration for page {page.page_number}">'
            )
        if audio:
            parts.append(
                f'<audio controls src="{file_base}/{escape(str(audio))}"></audio>'
            )
        parts.append("</div>")
        sections.append("".join(parts))

    extras: list[str] = []
    story_audio = media.get("story_audio")
    if story_audio:
        extras.append(
            "<h2>Listen to the whole book</h2>"
            f'<audio controls src="{file_base}/{escape(str(story_audio))}"></audio>'
        )
    video = media.get("video")
    if video:
        extras.append(
            "<h2>Watch it</h2>"
            f'<video controls src="{file_base}/{escape(str(video))}"></video>'
        )
    pdf = media.get("pdf")
    if pdf:
        extras.append(
            f'<p><a href="{file_base}/{escape(str(pdf))}">Download the PDF</a></p>'
        )

    directory = (
        f'<p class="status">Files: {escape(str(run_directory))}</p>'
        if run_directory
        else ""
    )

    body = (
        f"<h1>{escape(outline.title)}</h1>"
        f'<p class="subtitle">{escape(outline.logline)}</p>'
        f"{''.join(sections)}"
        f"{''.join(extras)}"
        f"{directory}"
        f'<p><a href="{back_link[0]}">{escape(back_link[1])}</a></p>'
    )
    return _document(f"{outline.title} - SparkStory", body)
