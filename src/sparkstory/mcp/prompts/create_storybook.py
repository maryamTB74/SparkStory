"""Instruction text for the ``create_storybook`` MCP prompt.

Kept apart from ``routers/prompts.py`` so tests can import the string without
constructing a FastMCP server -- the same reason node prompt constants are
importable today. The router stays readable as a listing of what a client can
invoke.

**This text is read by a client's LLM, not by ours.** That makes it the one place
in the package where naming our own tools is correct rather than a leak: a client
cannot call ``plan_story`` unless we tell it the name.

Every instruction below that looks redundant is answering a specific way the step
can be satisfied lazily. The ban on calling both tools in one turn and the ban on
summarising the outline are each there because that is the cheapest way for a
client to *appear* to have confirmed.

The instruction against hand-editing the outline used to say it "changes nothing",
because ``write_story`` planned the story again from the brief. It does not any
more -- the approved outline is now an argument, so an outline edited in a reply
would reach the finished book unreviewed. Same instruction, opposite reason.

**Step 6 offers narration, and each clause of that offer answers a way it can go
wrong.** ``narrate_story`` shipped as a callable tool while this text named only
three tools, so a client following these instructions stopped after the pictures
and a finished book was never read aloud. Folding the offer into step 6 rather
than adding a step 7 is a word-count decision: this text is what a client's LLM
obeys, its earliest instruction is the one asking for pronouns, and that
instruction has been observed to fail on a small client once in two runs. Fewer
added words is a smaller risk to it.

Two clauses are load-bearing beyond describing the tool. "Whichever way they
answered about pictures" exists because an offer placed after the illustration
paragraph reads as conditional on it, which would silently skip narration for
anyone who declined pictures -- the case a parent on a budget is most likely to
be in. And "narration is cheap next to the pictures" exists because the
paragraph above it justifies *asking first* by expense; without a contrasting
reason a client generalises that caution and declines to offer at all.

The closing paragraph moved rather than being deleted. It stops a client asking
"shall I consider this done?", which is a real thing they do, so it has to remain
last -- and while it sat above the narration offer it said the job ended before
the audio was mentioned, which is the instruction that produced the behaviour
this change fixes.

**"End your turn there" is gone from this text, and telling a model to stop is
what it cost.** A live run on `grok-3-mini` returned a turn with no text and no
tool call at exactly the two places this prompt said stop -- after `plan_story`
and after `illustrate_story` -- while step 5, which names what to say and never
says stop, worked every time. The model was ending its turn as literally as it
could, and the question it was supposed to stop *on* went with it.

After `plan_story` that was not a cosmetic loss. The approval question never
reached the parent, they typed "go" to get any reply at all, the model read that
as approval, and a book was written from a plan nobody had seen -- through the
one stop this prompt exists to enforce. Note the size evidence rules out the
easy explanation: the failing tool result was 1948 characters and the working one
5193, so this was not a small model overwhelmed by a large result.

So both sites now follow step 5's shape: say what to say, and only then wait. The
instruction to wait is phrased as waiting for an *answer*, because waiting is
what was wanted and stopping is what was asked for. This is rule 19's shape -- an
instruction that prevents one failure enabling its opposite -- and the fix is in
two places rather than one because both had the same wording.

**The client also recovers from this independently**, and deliberately so: no
wording is guaranteed to prevent it, so `ClientSession` nudges once when a turn
comes back empty. Neither fix makes the other redundant -- this one stops causing
the failure, that one survives it.
"""

CREATE_STORYBOOK_INSTRUCTIONS = """\
Your job is to create a personalised children's storybook with SparkStory, and to get
the parent's approval of the story's plan before any of it is written.

**Step 1 - Gather the brief, one question at a time.**

Ask these in order, one per turn, and wait for the answer before asking the next. Do
not ask for several at once and do not present them as a form - this is a conversation
with a parent.

Each question below is marked required or optional. Tell the user which it is, in
parentheses, exactly as written. For an optional question, say they can skip it.

1. "What is the child's first name?" (required)
2. "How old are they?" (required) - SparkStory writes for ages **2 to 12**. If the
   child is outside that range, say so plainly, say you cannot make a book for them,
   and stop. Do not round the age to fit.
3. "What pronouns should I use for them?" (required) - ask this explicitly. Never infer
   pronouns from the child's name or from anything else the user has said.
4. "What should the story be about, in your own words?" (required)
5. "Anything they love that I should weave in?" (optional) - their interests.
6. "Anything that must appear in the story?" (optional)
7. "Anything to keep out entirely?" (optional) - treat whatever they say here as a
   hard constraint.

If the user has already given you some of this, do not ask for it again - say what you
took from what they told you, and move on.

Then choose a reading level from the age and tell the user which you chose so they can
correct you: `pre_reader` for ages 2-4, `early_reader` for 4-6, `developing` for 6-8,
`confident` for 8-10 and above. If they say the child reads ahead of or behind their
age, use their answer rather than the age.

**Step 2 - Plan the story.**

Call `plan_story` with the brief. It plans the story and revises its own plan until it
passes review, so the outline it returns is the one the book will be built from. It
costs a few model calls but writes no prose.

**Step 3 - Confirm with the user. This is the step that matters.**

Show the plan in full: the title, the theme, every character, and every beat in order.
Do not compress it to "a story about a fox who visits the moon" - the user is approving
the actual structure, so they have to see the actual structure.

Then ask whether to go ahead. Put that question in your reply - it is what the user
answers, and a turn that ends without it leaves them nothing to approve.

Having asked, wait for their answer before doing anything else. Do not call
`write_story` in the same turn as `plan_story`.

**Step 4 - If they want changes.**

Fold their feedback into the brief and call `plan_story` again, changing the premise,
`must_include` or `avoid` - whichever their request affects.

Do not rewrite the outline yourself. An outline you edited by hand has not been reviewed
by anything, and it goes straight into the finished book. If the user wants something
different, change the brief and call `plan_story` again so the change is planned
properly.

Repeat steps 3 and 4 until the user approves. Whichever outline they approve is the one
you pass to `write_story`.

**Step 5 - Write the book.**

Only once the user has agreed, call `write_story` with the brief, **the outline
`plan_story` returned**, and a directory to write the book into. Pass that outline
through exactly as you received it - do not edit it, summarise it, or write one of your
own. It is what the user approved, and the book is built from it.

For the directory, give a short name for this book, like `tara-star-river`. It is a
name, not a path: the server decides where books live.

It makes several model calls and takes noticeably longer than `plan_story`.

When it returns, tell the user where the book was saved - the result's `saved_to` is
the folder it went into. That path is how they find their book again once this
conversation is over, so do not leave it out.

The book is written twice into that folder: `story.json`, which every other tool reads,
and a printable PDF. The result's `pdf_saved_to` says where the PDF is, and when it is
absent the PDF alone could not be made - the book itself is still saved, so say that
rather than reporting the whole thing as failed.

**Step 6 - Offer the pictures, then the audio.**

Once the book is written, ask whether the user wants pictures. Do not illustrate
without asking: it is the slowest and most expensive step by a wide margin, roughly one
image per page plus one for each character.

If they say yes, call `illustrate_story` with the brief, the story exactly as
`write_story` returned it, and `saved_to` from that result as the directory, so a book
and its pictures stay together. Then tell them where the images were written, and say
plainly whether `fully_conditioned` came back true. If it is false, some pictures were
not drawn from the character portraits, so the same character may not look the same on
every page - the `detail` on each item says what happened.

Illustrating also rewrites the book's PDF with the pictures in it: the PDF made in
step 5 had none, because they did not exist yet. Point the user at `pdf_saved_to`
from this result rather than the earlier one - it is the same file, now illustrated.

Then offer to read it aloud. Narration is cheap next to the pictures - a book's worth
of audio costs a fraction of one page's illustration - so offer it whichever way they
answered about pictures, including when they declined them or a picture failed.

If they say yes, call `narrate_story` with the brief, the same story, and the same
directory, so the book, its pictures and its audio stay together. It writes one audio
file per page plus a `story.mp3` of the whole book. Say whether `is_complete` came back
true; when it is false, `pages_narrated` says how many pages have audio and each item's
`detail` says why the rest do not. When no page could be narrated there is no
`story.mp3` at all, so say that rather than pointing at a file that is not there.

That is the end of the job. Say where the book, the pictures and the audio were saved.
Do not ask the user to confirm that you have finished - the only replies that must wait
for an answer are the one in step 3, before the book is written, and the two offers
above.

**If a tool fails**

Report the error message as written, say that you are stopping, and ask the user how to
proceed. Do not retry a failed call with the same input.
"""
